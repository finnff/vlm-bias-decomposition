from clip_utils import load_celeba_dataset, get_labels, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, eval_classifier, gender_classifier_accuracy
from embedding_analysis import pca_eval, tsne_eval, plot_attribute_bias_directions
from misclassified_vis import show_misclassified
from attribute_bias_analysis import compare_male_groups_ttest


def main():
    dataset = load_celeba_dataset(root="./data", split="train", download=False)

    clip_embeddings, gender_labels, images = get_labels(dataset, batch_size=64, max_samples=5000)

    clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices = train_gender_classifier(clip_embeddings, gender_labels)

    gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)

    eval_classifier(y_test, y_pred)

    pca_eval(clip_embeddings, gender_labels)
    tsne_eval(clip_embeddings, gender_labels)

    clip_embs_all, attr_mat, idx_list, attr_names = get_all_embeddings_and_attrs(dataset, batch_size=64, max_samples=5000)

    plot_attribute_bias_directions(clip_embs_all, attr_mat, attr_names)

    show_misclassified(images, y_test, y_pred, misclassified_indices, n=5)


    compare_male_groups_ttest(clf, clip_embeddings, attr_mat, attr_names)



if __name__ == "__main__":
    main()
