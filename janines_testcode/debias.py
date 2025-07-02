import numpy as np
import torch
from clip_utils import get_labels, load_celeba_dataset, get_all_embeddings_and_attrs
from gender_classification import train_gender_classifier, gender_classifier_accuracy, eval_classifier
from attribute_bias_analysis import compare_groups_ttest
import io
import sys


bias_attribute_names = ["Attractive"]

def hard_debias(embeddings, bias_basis):
    """
    Remove the enitre component / embeddings

    subtract full projection P=BB^T
    """
    B = bias_basis.T  #D, k
    P = B @ B.T       #D, D projection onto bias space
    if isinstance(embeddings, np.ndarray):
        return embeddings - embeddings @ P
    else:
        return embeddings - embeddings @ P.to(embeddings.device)


def soft_debias(embeddings, bias_basis, lam=1.0):
    """
    making projection smaller of biased vectors.

    math:
    min_{x'} ||x-x'||^2+lam * ||Proj_B(x')||^2

    """
    B = bias_basis.T        #D, k
    P = B @ B.T             #D, D
    alpha = lam / (1.0 + lam)  #how much smaller param
    if isinstance(embeddings, np.ndarray):
        return embeddings - alpha * (embeddings @ P)
    else:
        return embeddings - alpha * (embeddings @ P.to(embeddings.device))

print("0")
dataset = load_celeba_dataset(root="./data", split="train", download=False)
print("1")
clip_embeddings, gender_labels, images = get_labels(dataset, batch_size=64, max_samples=5000)
print("2")
X = clip_embeddings.numpy() if torch.is_tensor(clip_embeddings) else clip_embeddings
y = gender_labels.numpy() if torch.is_tensor(gender_labels) else gender_labels
print("3")
clip_embs, attr_mat, idx_list, attr_names = get_all_embeddings_and_attrs(dataset,max_samples=5000)
print("4")

bias_vectors = []
for attr in bias_attribute_names:
    idx = attr_names.index(attr)
    mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
    mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
    v = mu_pos - mu_neg
    v = v.astype(np.float32)
    v /= np.linalg.norm(v)
    bias_vectors.append(v)


B = np.stack(bias_vectors, axis=1)
Q, _ = np.linalg.qr(B)
bias_basis = Q.T                              #(2, D)

X_hard_yt = hard_debias(X, bias_basis)
X_soft_yt = soft_debias(X, bias_basis, lam=0)

clf, X_train, X_test, y_train, y_test, y_pred, misclassified_indices = train_gender_classifier(X_soft_yt, gender_labels)

gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)
eval_classifier(y_test, y_pred)

compare_groups_ttest(clf, clip_embeddings, attr_mat, attr_names, bias_attribute_names)

results = []

for attr in attr_names:
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()

    try:
        compare_groups_ttest(clf, clip_embeddings, attr_mat, attr_names, [attr])
        output = mystdout.getvalue()
        lines = output.strip().split("\n")

        t_line = [line for line in lines if "T-test result" in line]
        if t_line:
            t_str = t_line[0]
            t_val = float(t_str.split("t =")[1].split(",")[0].strip())
            p_val = float(t_str.split("p =")[1].strip())
            sig = "*" if p_val < 0.05 else ""
            output += f"Significant: {sig}\n"

        results.append(output)

    except Exception as e:
        print(f"Error on attribute {attr}: {e}")

    finally:
        sys.stdout = old_stdout

print("\n" + "="*60)
print("T-TEST SUMMARY FOR ALL ATTRIBUTES (p < 0.05 marked with *)")
print("="*60)

for r in results:
    print("\n" + r)




bias_attribute_names_sig = [
    "5_o_Clock_Shadow", "Bald", "Bangs", "Goatee", "High_Cheekbones",
    "Mustache", "No_Beard", "Rosy_Cheeks", "Sideburns", "Straight_Hair",
    "Wearing_Lipstick", "Wearing_Necktie"
]

def find_optimal_lambda(attr, X, attr_mat, clip_embeddings, attr_names, gender_labels, lambdas=np.linspace(0.0, 10.0, 30)):
    idx = attr_names.index(attr)
    
    # Create single bias vector
    mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
    mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
    v = mu_pos - mu_neg
    v = v.astype(np.float32)
    
    if np.linalg.norm(v) < 1e-6:
        return None, None, None, None, "[Skipped: degenerate vector]"

    v /= np.linalg.norm(v)
    B = v.reshape(-1, 1)
    Q, _ = np.linalg.qr(B)
    bias_basis = Q.T  # shape [1, D]

    best_lambda = None
    best_alpha = None
    best_t = float("inf")
    best_p = None
    best_acc = None
    best_output = ""

    for lam in lambdas:
        alpha = lam / (1.0 + lam)
        X_soft = soft_debias(X, bias_basis, lam=lam)

        clf, X_train, X_test, y_train, y_test, y_pred, _ = train_gender_classifier(X_soft, gender_labels)
        acc = gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)

        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            compare_groups_ttest(clf, torch.tensor(X_soft), attr_mat, attr_names, [attr])
            output = mystdout.getvalue()
            lines = output.strip().split("\n")
            t_line = [line for line in lines if "T-test result" in line]
            if t_line:
                t_val = float(t_line[0].split("t =")[1].split(",")[0].strip())
                p_val = float(t_line[0].split("p =")[1].strip())
                if abs(t_val) < abs(best_t):
                    best_lambda = lam
                    best_alpha = alpha
                    best_t = t_val
                    best_p = p_val
                    best_acc = acc
                    best_output = output
        except Exception as e:
            print(f"[ERROR] Attr: {attr} λ={lam}: {e}")
        finally:
            sys.stdout = old_stdout

    return best_alpha, best_t, best_p, best_acc, best_output


results = []
for attr in bias_attribute_names_sig:
    print(f"\n--- Optimizing λ for: {attr} ---")
    best_alpha, best_t, best_p, best_acc, best_output = find_optimal_lambda(attr, X, attr_mat, clip_embeddings, attr_names, gender_labels)

    
    if best_alpha is not None:
        results.append(
    f"Attribute: {attr}\n"
    f"Best α (λ / (1+λ)): {best_alpha:.3f}\n"
    f"T-statistic: {best_t:.3f},  p = {best_p:.2e},  Accuracy = {best_acc:.4f}\n\n"
    f"{best_output}"
)
    else:
        results.append(f"Attribute: {attr}\n[ERROR] No valid λ found\n")



print("\n" + "="*60)
print("TUNED DEBIASING RESULTS FOR BIASED ATTRIBUTES")
print("="*60)
for r in results:
    print("\n" + r)



print("\n HARD DEBIASING")

hard_debias_results = []

for attr in bias_attribute_names_sig:  # your chosen significant attribute list
    idx = attr_names.index(attr)
    mu_pos = X[attr_mat[:, idx] == 1].mean(axis=0)
    mu_neg = X[attr_mat[:, idx] == 0].mean(axis=0)
    v = mu_pos - mu_neg
    v = v.astype(np.float32)
    v /= np.linalg.norm(v)
    bias_basis = v[None, :]  # shape (1, D)

    X_hard = hard_debias(X, bias_basis)
    clf, X_train, X_test, y_train, y_test, y_pred, _ = train_gender_classifier(X_hard, y)
    acc = gender_classifier_accuracy(clf, X_train, y_train, X_test, y_test)

    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        compare_groups_ttest(clf, torch.tensor(X_hard), attr_mat, attr_names, [attr])
        output = mystdout.getvalue()
        lines = output.strip().split("\n")

        t_line = [line for line in lines if "T-test result" in line]
        t_val, p_val = None, None
        if t_line:
            t_str = t_line[0]
            t_val = float(t_str.split("t =")[1].split(",")[0].strip())
            p_val = float(t_str.split("p =")[1].strip())

        hard_debias_results.append({
            "attribute": attr,
            "accuracy": acc,
            "t_stat": t_val,
            "p_val": p_val
        })

    except Exception as e:
        print(f"Error with attribute {attr}: {e}")
    finally:
        sys.stdout = old_stdout

# Print summary
print("\n{:<20} | {:<10} | {:<8} | {}".format("Attribute", "Accuracy", "t-stat", "p"))
print("-" * 60)
for res in hard_debias_results:
    print(f"{res['attribute']:<20} | {res['accuracy']:.4f}    | {res['t_stat']:.2f}    | {res['p_val']:.3e}")