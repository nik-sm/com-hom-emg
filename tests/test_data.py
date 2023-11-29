from com_hom_emg.data import get_datasets, get_per_subj_data


def test_get_datasets_disjoint_val_test():
    # The subject used for val should be different each time
    # Likewise for test
    per_subj_data = get_per_subj_data()
    all_val_subj = []
    all_test_subj = []
    for i in range(10):
        _train_set, _val_set, _test_set, _train_subj, val_subj, test_subj = get_datasets(
            per_subj_data=per_subj_data,
            fold=i,
            n_train_subj=8,
            n_val_subj=1,
            n_test_subj=1,
            use_preprocessed_data=False,
            return_subj_names=True,
        )
        all_val_subj.append(val_subj[0])
        all_test_subj.append(test_subj[0])

    assert len(set(all_val_subj)) == len(all_val_subj)
    assert len(set(all_test_subj)) == len(all_test_subj)
