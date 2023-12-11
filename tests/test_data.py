import torch

from com_hom_emg.data import get_datasets, get_per_subj_data


def test_get_datasets_disjoint_val_test():
    # The subject used for val should be different each time
    # Likewise for test
    per_subj_data = get_per_subj_data()
    all_val_subj = []
    all_test_subj = []

    n_train = 8
    n_val = 1
    n_test = 1

    expected_train_size = 8 * 1224  # 1224 gestures per subject
    expected_val_size = n_val * 1224
    expected_test_size = n_test * 1224

    def check_contents(dataset, N):
        ## Check shapes
        # data =  8-channel EMG, 962 timesteps (= 500ms at 1926 Hz)
        assert dataset.tensors[0].shape == torch.Size([N, 8, 962])
        # labels = 2D labels
        assert dataset.tensors[1].shape == torch.Size([N, 2])
        # is_single = bool labels
        assert dataset.tensors[2].shape == torch.Size([N])
        # subj_ids = 1d labels
        assert dataset.tensors[3].shape == torch.Size([N])

        ## Check dtypes
        assert dataset.tensors[0].dtype == torch.float32
        assert dataset.tensors[1].dtype == torch.int64
        assert dataset.tensors[2].dtype == torch.bool
        assert dataset.tensors[3].dtype == torch.int64

    for i in range(10):
        train_set, val_set, test_set, train_subj, val_subj, test_subj = get_datasets(
            per_subj_data=per_subj_data,
            fold=i,
            n_train_subj=n_train,
            n_val_subj=n_val,
            n_test_subj=n_test,
            use_preprocessed_data=False,
            return_subj_names=True,
        )
        all_val_subj.append(val_subj[0])
        all_test_subj.append(test_subj[0])

        assert len(train_subj) == n_train
        assert len(val_subj) == n_val
        assert len(test_subj) == n_test

        check_contents(train_set, expected_train_size)
        check_contents(val_set, expected_val_size)
        check_contents(test_set, expected_test_size)

    assert len(set(all_val_subj)) == len(all_val_subj)
    assert len(set(all_test_subj)) == len(all_test_subj)
