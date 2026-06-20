import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=99, num_features_xt=25,
                 num_features_mol=8, n_filters=32, embed_dim=128,
                 output_dim=128, protein_feat_dim=0,
                 protein_feat_hidden=64, klifs_hidden=64,
                 gatekeeper_hidden=16, hinge_hidden=32, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

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
        self.protein_context = nn.Linear(self.protein_context_dim, output_dim)
        self.protein_context_norm = nn.LayerNorm(self.protein_context_dim)
        self.ligand_gate = nn.Linear(output_dim, output_dim)

        self.fc1 = nn.Linear(
            output_dim + output_dim + num_features_mol +
            protein_feat_hidden + klifs_hidden +
            gatekeeper_hidden + hinge_hidden,
            1024
        )
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
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

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        if node_mask is not None:
            x = x * node_mask
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        if node_mask is not None:
            x = x * node_mask
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        if node_mask is not None:
            x = x * node_mask
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        if node_mask is not None:
            x = x * node_mask
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        if node_mask is not None:
            x = x * node_mask
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        embedded_klifs = self.embedding_xt(klifs_pocket)
        conv_klifs = self.conv_klifs_1(embedded_klifs)
        kp = conv_klifs.view(-1, 32 * 121)
        kp = self.fc1_klifs(kp)
        kp = self.relu(kp)
        kp = self.dropout(kp)

        embedded_gatekeeper = self.embedding_xt(gatekeeper).view(batch_size, -1)
        gk = self.fc_gatekeeper(embedded_gatekeeper)
        gk = self.relu(gk)
        gk = self.dropout(gk)

        embedded_hinge = self.embedding_xt(hinge).view(batch_size, -1)
        hg = self.fc_hinge(embedded_hinge)
        hg = self.relu(hg)
        hg = self.dropout(hg)

        pf = self.fc_protein_feat(protein_feat)
        pf = self.relu(pf)
        pf = self.dropout(pf)

        protein_ctx = torch.cat((xt, pf, kp, gk, hg), 1)
        protein_ctx = self.protein_context_norm(protein_ctx)
        protein_ctx = self.protein_context(protein_ctx)
        protein_ctx = self.relu(protein_ctx)
        protein_ctx = self.dropout(protein_ctx)

        gate = torch.sigmoid(self.ligand_gate(protein_ctx))
        x = x * gate

        xc = torch.cat((x, protein_ctx, mol_features, pf, kp, gk, hg), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
