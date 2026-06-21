import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,
                 num_features_xd=99, num_features_xt=25,
                 num_features_mol=8, output_dim=128,
                 protein_feat_dim=0, protein_feat_hidden=64,
                 klifs_hidden=64, gatekeeper_hidden=16,
                 hinge_hidden=32, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(
            in_channels=1000,
            out_channels=n_filters,
            kernel_size=8
        )
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # additional molecular and protein features
        self.norm_mol = nn.LayerNorm(num_features_mol)

        self.norm_protein_feat = nn.LayerNorm(protein_feat_dim)
        self.fc_protein_feat = nn.Linear(protein_feat_dim, protein_feat_hidden)

        self.conv_klifs_1 = nn.Conv1d(
            in_channels=85,
            out_channels=n_filters,
            kernel_size=8
        )
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

        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
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

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        if node_mask is not None:
            x = x * node_mask

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        if node_mask is not None:
            x = x * node_mask

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        if node_mask is not None:
            x = x * node_mask

        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

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