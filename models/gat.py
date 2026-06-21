import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=99, n_output=1, num_features_xt=25,
                 num_features_mol=8, n_filters=32, embed_dim=128,
                 output_dim=128, protein_feat_dim=0,
                 protein_feat_hidden=64, klifs_hidden=64,
                 gatekeeper_hidden=16, hinge_hidden=32, dropout=0.2):

        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # additional molecular and protein features
        self.norm_mol = nn.LayerNorm(num_features_mol)
        self.norm_protein_feat = nn.LayerNorm(protein_feat_dim)
        self.fc_protein_feat = nn.Linear(protein_feat_dim, protein_feat_hidden)

        self.conv_klifs_1 = nn.Conv1d(in_channels=85, out_channels=n_filters, kernel_size=8)
        self.fc1_klifs = nn.Linear(32 * 121, klifs_hidden)

        self.fc_gatekeeper = nn.Linear(embed_dim, gatekeeper_hidden)
        self.fc_hinge = nn.Linear(3 * embed_dim, hinge_hidden)

        self.protein_context_dim = (
            output_dim + protein_feat_hidden + klifs_hidden +
            gatekeeper_hidden + hinge_hidden
        )

        self.protein_context_norm = nn.LayerNorm(self.protein_context_dim)
        self.protein_context = nn.Linear(self.protein_context_dim, output_dim)
        self.ligand_gate = nn.Linear(output_dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(
            output_dim + output_dim + num_features_mol +
            protein_feat_hidden + klifs_hidden +
            gatekeeper_hidden + hinge_hidden,
            1024
        )

        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_output = n_output

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        batch_size = target.size(0)

        node_mask = getattr(data, "node_mask", None)

        mol_features = data.mol_features.view(batch_size, -1)
        mol_features = self.norm_mol(mol_features)

        protein_feat = data.protein_feat.view(batch_size, -1)
        protein_feat = self.norm_protein_feat(protein_feat)

        klifs_pocket = data.klifs_pocket.view(batch_size, -1)
        gatekeeper = data.gatekeeper.view(batch_size, -1)
        hinge = data.hinge.view(batch_size, -1)

        if node_mask is not None:
            x = x * node_mask

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))

        if node_mask is not None:
            x = x * node_mask

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        if node_mask is not None:
            x = x * node_mask

        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # protein input feed-forward:
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # KLIFS pocket input feed-forward
        embedded_klifs = self.embedding_xt(klifs_pocket)
        conv_klifs = self.conv_klifs_1(embedded_klifs)
        kp = conv_klifs.view(-1, 32 * 121)
        kp = self.fc1_klifs(kp)
        kp = self.relu(kp)
        kp = self.dropout(kp)

        # gatekeeper input feed-forward
        embedded_gatekeeper = self.embedding_xt(gatekeeper).view(batch_size, -1)
        gk = self.fc_gatekeeper(embedded_gatekeeper)
        gk = self.relu(gk)
        gk = self.dropout(gk)

        # hinge input feed-forward
        embedded_hinge = self.embedding_xt(hinge).view(batch_size, -1)
        hg = self.fc_hinge(embedded_hinge)
        hg = self.relu(hg)
        hg = self.dropout(hg)

        # additional protein feature input feed-forward
        pf = self.fc_protein_feat(protein_feat)
        pf = self.relu(pf)
        pf = self.dropout(pf)

        # protein context
        protein_ctx = torch.cat((xt, pf, kp, gk, hg), 1)
        protein_ctx = self.protein_context_norm(protein_ctx)
        protein_ctx = self.protein_context(protein_ctx)
        protein_ctx = self.relu(protein_ctx)
        protein_ctx = self.dropout(protein_ctx)

        # gate ligand representation with protein context
        gate = torch.sigmoid(self.ligand_gate(protein_ctx))
        x = x * gate

        # concat
        xc = torch.cat((x, protein_ctx, mol_features, pf, kp, gk, hg), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out