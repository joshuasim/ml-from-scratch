# ML From Scratch

Implementing classic machine learning algorithms using only NumPy — no sklearn.

The goal isn't to build production-ready libraries. It's to understand each algorithm well enough to derive it, explain it, and debug it without looking anything up. Every implementation starts from the math(statquest youtube videos), gets coded from memory, and gets tested against sklearn to verify correctness using a dummy data set.

---

## Algorithms


| #   | Algorithm                     | Status                                                    | File                     |
| --- | ----------------------------- | --------------------------------------------------------- | ------------------------ |
| 1   | Linear Regression             | ✅ Complete                                               | `linear_regression.py`   |
| 2   | Logistic Regression           | ✅ Complete                                               | `logistic_regression.py` |
| 3   | Decision Tree (depth-1 stump) | ✅ Complete                                               | `decision_tree.py`       |
| 4   | Naive Bayes                   | ✅ Complete                                               | `naive_bayes.py`         |
| 5   | K-Means Clustering            | 🔜 Coming soon                                            | —                        |
| 6   | KNN                           | 🔜 Coming soon                                            | —                        |
| 7   | MLP Neural Network            | 🔜 Coming soon                                            | —                        |
| 8   | PCA                           | 🔜 Coming soon                                            | —                        |
| 9   | Random Forest                 | 🔜 Coming soon                                            | —                        |


---

## Approach

For each algorithm:

1. I watch relevant StatQuest videos to understand the math
2. Implement from scratch in NumPy from memory
3. Test output against sklearn on a dummy dataset
4. Write a plain-English explanation of how the algorithm works as if explaining it in an interview

Each file is commented to explain not just what the code does but why — the math behind each step, where the gradients come from, and what would break if you did it differently.

---

## What's not here

These are intentionally minimal implementations focused on the core algorithm. They don't include regularization, multiclass extensions, or production-level error handling. Those come later — in the portfolio projects where they solve a real problem.

---

## Reference

After attempting each implementation, I check my work against [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) to identify gaps. The implementations here are written independently before consulting that repo.