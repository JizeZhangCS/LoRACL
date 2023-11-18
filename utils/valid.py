def cls_metrics(test_features, test_labels, cls):
    acc1 = cls.score(test_features, test_labels)
    return acc1, 0, 0, 0, 0, 0, 0, 0

def ret_metrics(test_features, test_labels, train_labels, nearest_neigh):
    _, top_n_matches_ids = nearest_neigh.kneighbors(test_features)
    top_n_labels = train_labels[top_n_matches_ids]
    correct = test_labels[:, None] == top_n_labels

    acc1 = correct[:, 0:1].any(-1).mean()
    acc10 = correct[:, 0:10].any(-1).mean()
    acc20 = correct[:, 0:20].any(-1).mean()

    p1 = correct[:, 0:1].sum(-1).mean()
    p10 = correct[:, 0:10].sum(-1).mean() / 10
    p20 = correct[:, 0:20].sum(-1).mean() / 20
    p5 = correct[:, 0:5].sum(-1).mean() / 5
    p15 = correct[:, 0:15].sum(-1).mean() / 15
    return acc1, acc10, acc20, p1, p5, p10, p15, p20