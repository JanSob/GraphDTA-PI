import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA


class TestbedDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for drug-target affinity samples.

    Loads preprocessed ligand/protein graph data when available, or builds
    and saves it from SMILES, encoded protein inputs, and affinity labels.
    """
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, xp=None, xk=None, xg=None, xh=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xp, xk, xg, xh, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xp, xk, xg, xh, y, smile_graph):
        assert (
            len(xd) == len(xt)
            and len(xt) == len(xp)
            and len(xp) == len(xk)
            and len(xk) == len(xg)
            and len(xg) == len(xh)
            and len(xh) == len(y)
        ), "All input lists must be the same length!"

        data_list = []
        data_len = len(xd)

        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))

            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            protein_feat = xp[i]
            klifs_pocket = xk[i]
            gatekeeper = xg[i]
            hinge = xh[i]

            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index, mol_features = smile_graph[smiles]

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels])
            )

            GCNData.target = torch.LongTensor([target])
            GCNData.mol_features = torch.FloatTensor(mol_features).view(1, -1)
            GCNData.protein_feat = torch.FloatTensor([protein_feat])
            GCNData.klifs_pocket = torch.LongTensor([klifs_pocket])
            GCNData.gatekeeper = torch.LongTensor([gatekeeper])
            GCNData.hinge = torch.LongTensor([hinge])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)

        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])