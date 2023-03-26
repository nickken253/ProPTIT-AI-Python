# AI CƠ BẢN - BUỔI 1
Câu hỏi kỳ này:
- Machine Learning là gì.
- Phân nhóm các thuật toán Machine Learning.
- Python Syntax. (mức cơ bản).
## 1. Machine Learning là gì?
- Machine Learning là một tập con, một phương pháp của trí tuệ nhân tạo (AI) cho phép máy tính tự động học hỏi từ dữ liệu mà không cần được lập trình một cách cụ thể. Nó giúp máy tính học hỏi và cải thiện hiệu suất của chúng theo thời gian, không cần được lập trình để giải quyết một vấn đề cụ thể.
- Các thuật toán học máy sử dụng các mô hình toán học để phân tích dữ liệu và tìm ra các quy luật hoặc mô hình ẩn sau đó áp dụng chúng để dự đoán hoặc phân loại các dữ liệu mới. Các thuật toán này có thể được sử dụng để giải quyết nhiều vấn đề khác nhau, chẳng hạn như nhận dạng ảnh, dịch thuật tự động, phân loại email spam, dự báo thời tiết, phân tích tín hiệu vô tuyến, và nhiều ứng dụng khác.
## 2. Phân nhóm các thuật toán theo phương thức học.
- Machine learning được chia thành ba loại chính:  
    + Học giám sát (supervised learning): dữ liệu được cung cấp kèm theo các nhãn (labels) để giúp thuật toán học hỏi.
    + Học không giám sát (unsupervised learning): dữ liệu không được cung cấp nhãn và thuật toán phải tự tìm ra cấu trúc trong dữ liệu
    + Học bán giám sát (semi-supervised learning): một phần dữ liệu được cung cấp nhãn và một phần không được cung cấp nhãn
    + Học củng cố / tăng cường (reinforcement learning): thuật toán học hỏi thông qua việc tương tác với môi trường.
### 2.1. Học giám sát (supervised learning)
- Học giám sát là một loại học máy trong đó thuật toán học được huấn luyện trên một tập dữ liệu được gán nhãn. Ví dụ: nhận dạng ảnh, phân loại email spam, dự đoán giá nhà, dự đoán thời tiết, phân tích tín hiệu vô tuyến, và nhiều ứng dụng khác.
- Supervised learning là thuật toán dự đoán đầu ra (outcome) của một dữ liệu mới (new input) dựa trên các cặp (input, outcome) đã biết từ trước. Cặp dữ liệu này còn được gọi là (data, label). Supervised learning là nhóm phổ biến nhất trong các thuật toán Machine Learning.
- Một cách toán học, Supervised learning là khi chúng ra có một tập hợp biến đầu vào $X = { x_1 , x_2 , … , x_N }$ và một tập hợp nhãn tương ứng $Y = { y_1 , y_2 , … , y_N }$ , trong đó $x_i$ , $y_i$ là các vector. Các cặp dữ liệu biết trước ( $x_i$ , $y_i$ ) $∈$ $X×Y$ được gọi là tập training data (dữ liệu huấn luyện). Từ tập training data này, chúng ta cần tạo ra một hàm số ánh xạ mỗi phần tử từ tập $X$ sang một phần tử (xấp xỉ) tương ứng của tập $Y$:
    > $y_i ≈ f(x_i), ∀i=1,2,…,N$
- Ví dụ, trong bài toán phân loại hình ảnh, tập dữ liệu được gắn nhãn có thể bao gồm các hình ảnh của các đối tượng khác nhau (chó, mèo, người, xe hơi, v.v.). Mỗi hình ảnh được gắn nhãn với một nhãn tương ứng với đối tượng trong hình ảnh đó. Mục tiêu của thuật toán học có giám sát là học được một mô hình có thể phân loại các hình ảnh không được gắn nhãn thành các đối tượng khác nhau.
- `Ví dụ 1`: trong nhận dạng chữ viết tay, ta có ảnh của hàng nghìn ví dụ của mỗi chữ số được viết bởi nhiều người khác nhau. Chúng ta đưa các bức ảnh này vào trong một thuật toán và chỉ cho nó biết mỗi bức ảnh tương ứng với chữ số nào. Sau khi thuật toán tạo ra một mô hình, tức một hàm số mà đầu vào là một bức ảnh và đầu ra là một chữ số, khi nhận được một bức ảnh mới mà mô hình chưa nhìn thấy bao giờ, nó sẽ dự đoán bức ảnh đó chứa chữ số nào.
- `Ví dụ 2`: Thuật toán dò các khuôn mặt trong một bức ảnh đã được phát triển từ rất lâu. Thời gian đầu, facebook sử dụng thuật toán này để chỉ ra các khuôn mặt trong một bức ảnh và yêu cầu người dùng tag friends - tức gán nhãn cho mỗi khuôn mặt. Số lượng cặp dữ liệu (khuôn mặt, tên người) càng lớn, độ chính xác ở những lần tự động tag tiếp theo sẽ càng lớn.

#### 2.1.1. Các thuật toán học giám sát
- Classification (phân loại) được sử dụng để phân loại các điểm dữ liệu vào các nhóm đã được `xác định trước` đó. Thuật toán này xây dựng một mô hình từ các dữ liệu đã biết và sử dụng mô hình này để phân loại các dữ liệu mới vào các nhóm đã biết. 
    + Ví dụ, một ứng dụng phân loại email có thể được xây dựng để phân loại các email vào hai nhóm: "spam" và "không phải spam". Mô hình phân loại sẽ được xây dựng từ các email đã được đánh dấu là spam hoặc không phải spam và sử dụng mô hình này để phân loại các email mới vào hai nhóm tương ứng. 
    + Các thuật toán phân loại phổ biến bao gồm Decision Trees, Naive Bayes, Support Vector Machines (SVM), và Neural Networks.
- Regression (hồi quy) là một trong những thuật toán quan trọng trong học máy, nó được sử dụng để dự đoán giá trị của một biến phụ thuộc dựa trên các biến độc lập khác. Regression có thể được sử dụng để giải quyết các vấn đề dự đoán như giá cổ phiếu, giá nhà, doanh số bán hàng, v.v. 
    + Trong regression, một mô hình được xây dựng bằng cách tìm một mối quan hệ giữa các biến độc lập và biến phụ thuộc, thông qua việc sử dụng các thuật toán thống kê. 
    + Các thuật toán regression phổ biến bao gồm Linear Regression, Polynomial Regression, Ridge Regression và Lasso Regression. Các mô hình hồi quy có thể được sử dụng để dự đoán giá trị của biến phụ thuộc cho các giá trị của các biến độc lập mới được cung cấp.

### 2.2. Học không giám sát (Unsupervised Learning)
- Trong thuật toán này, chúng ta không biết được outcome hay nhãn mà chỉ có dữ liệu đầu vào. Thuật toán unsupervised learning sẽ dựa vào cấu trúc của dữ liệu để thực hiện một công việc nào đó. Một số ứng dụng của học không giám sát bao gồm phân nhóm (clustering) dữ liệu, giảm chiều dữ liệu (dimensionality reduction), và tìm kiếm các mẫu hoặc luật trong dữ liệu. Ví dụ, thuật toán K-Means clustering là một thuật toán phổ biến trong học không giám sát được sử dụng để phân nhóm dữ liệu vào các nhóm tương tự nhau, dựa trên các đặc trưng chung của chúng. PCA (Principal Component Analysis) là một phương pháp giảm chiều dữ liệu phổ biến trong học không giám sát, được sử dụng để giảm số lượng chiều của dữ liệu bằng cách tìm các thành phần chính trong dữ liệu.
#### 2.2.1. Các thuật toán học không giám sát
- Clustering (phân nhóm) là một thuật toán phổ biến trong học không giám sát, nó được sử dụng để phân nhóm các điểm dữ liệu vào các nhóm khác nhau dựa trên các đặc trưng chung của chúng. 
    + Thuật toán này không yêu cầu các nhãn hoặc đầu ra mong đợi, mà chỉ phân nhóm các điểm dữ liệu dựa trên đặc trưng của chúng. Các thuật toán phân nhóm phổ biến bao gồm K-Means Clustering, Hierarchical Clustering và DBSCAN (Density-Based Spatial Clustering of Applications with Noise). 
    + Các phương pháp phân nhóm này có thể được sử dụng trong nhiều ứng dụng khác nhau, chẳng hạn như phân nhóm khách hàng hoặc phân tích hình thái của các tế bào trong y học.
- Association (kết hợp) là một phương pháp phân tích dữ liệu trong học không giám sát, được sử dụng để tìm kiếm các mối quan hệ giữa các mục trong tập dữ liệu. Cụ thể, phương pháp này tìm kiếm các luật kết hợp (association rules) giữa các mục trong tập dữ liệu, trong đó một mục có thể dẫn đến xuất hiện của một hoặc nhiều mục khác. 
    + Một ví dụ về phương pháp kết hợp là trong một tập dữ liệu bán lẻ, nếu khách hàng mua sản phẩm A thì khả năng cao họ cũng sẽ mua sản phẩm B. Bằng cách phân tích tập dữ liệu, phương pháp kết hợp có thể xác định các luật kết hợp như vậy và đưa ra các đề xuất khuyến nghị sản phẩm cho khách hàng. 
    + Các thuật toán phổ biến trong phương pháp kết hợp bao gồm Apriori và Eclat.
### 2.3. Học bán giám sát (Semi-supervised Learning)
- Học bán giám sát (Semi-supervised Learning) là một dạng học máy nằm giữa hai dạng chính: học có giám sát và học không giám sát. Học bán giám sát sử dụng một số lượng dữ liệu được gán nhãn (dữ liệu có nhãn) kết hợp với một số lượng dữ liệu không được gán nhãn (dữ liệu không có nhãn) để huấn luyện mô hình.
- Khi sử dụng học bán giám sát, một phần dữ liệu được gán nhãn được sử dụng để huấn luyện mô hình, trong khi phần còn lại được sử dụng để giúp mô hình tìm hiểu các đặc trưng quan trọng của dữ liệu. Việc sử dụng dữ liệu không được gán nhãn giúp mô hình học được các đặc trưng phổ biến trong dữ liệu và giúp cải thiện độ chính xác của mô hình.
- Học bán giám sát có thể được áp dụng trong nhiều lĩnh vực khác nhau, chẳng hạn như nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên và phân loại tài liệu. Các phương pháp học bán giám sát phổ biến bao gồm các phương pháp truyền thống như Self-Training, Co-Training và các phương pháp dựa trên mô hình như MixMatch, Tri-training và Noisy Student.

### 2.4. Học tăng cường (Reinforcement Learning)
- Học tăng cường (Reinforcement Learning) là một phương pháp học máy, trong đó một hệ thống học tập được huấn luyện thông qua việc tương tác với một môi trường và nhận phản hồi thông qua việc nhận được phần thưởng hoặc hình phạt. Mục tiêu của học tăng cường là tìm ra một chính sách (policy) tối ưu để đạt được một mục tiêu cụ thể.
- Trong học tăng cường, một hệ thống học tập (agent) được đưa vào một môi trường có trạng thái (state), một tập hành động (action) có thể thực hiện được và một hàm phần thưởng (reward) để đánh giá các hành động của hệ thống. Hệ thống học tập phải học cách tương tác với môi trường để đạt được mục tiêu, thông qua việc tìm kiếm một chính sách tối ưu để chọn hành động tốt nhất dựa trên trạng thái hiện tại của môi trường.
- Một ví dụ về học tăng cường là một robot học tập cách di chuyển trong một phòng để đến một đích. Robot sẽ đưa ra các hành động và nhận được phần thưởng hoặc hình phạt dựa trên trạng thái của nó trong phòng. Mục tiêu của robot là tìm ra một chính sách tối ưu để di chuyển đến đích một cách nhanh nhất và an toàn nhất.
- Các thuật toán phổ biến trong học tăng cường bao gồm Q-learning, SARSA và Deep Q-Network (DQN).
## 3. Phân loại dựa trên chức năng
- Có một cách phân nhóm thứ hai dựa trên chức năng của các thuật toán. Trong phần này, tôi xin chỉ liệt kê các thuật toán. Thông tin cụ thể sẽ được trình bày trong các bài viết khác tại blog này. Trong quá trình viết, tôi có thể sẽ thêm bớt một số thuật toán.


1. Regression Algorithms
- Linear Regression
- Logistic Regression
- Stepwise Regression

2. Classification Algorithms
- Linear Classifier
- Support Vector Machine (SVM)
- Kernel SVM
- Sparse Representation-based classification (SRC)

3. Instance-based Algorithms
- k-Nearest Neighbor (kNN)
- Learning Vector Quantization (LVQ)

4. Regularization Algorithms
- Ridge Regression
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Least-Angle Regression (LARS)

5. Bayesian Algorithms
- Naive Bayes
- Gaussian Naive Bayes

6. Clustering Algorithms
- k-Means clustering
- k-Medians
- Expectation Maximization (EM)

7. Artificial Neural Network Algorithms
- Perceptron
- Softmax Regression
- Multi-layer Perceptron
- Back-Propagation

8. Dimensionality Reduction Algorithms
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

9. Ensemble Algorithms
- Boosting
- AdaBoost
- Random Forest
Và còn rất nhiều các thuật toán khác.
## 4. Syntax Python cơ bản
- In ra một dòng chữ:
```python
print("Hello World")
```
- Gán giá trị cho một biến:
```python
a = 1   
b = 2
```
- Điều kiện if-else:
```python
if a > b:
    print("a > b")
else:
    print("a <= b")
```
- Vòng lặp for:
```python
for i in range(10):
    print(i)
```
- Vòng lặp while:
```python
while a < b:
    print(a)
    a += 1
```
- Hàm:
```python
def sum(a, b):
    return a + b
```
- Import thư viện:
```python
import numpy as np
```
- List (danh sách):
```python
a = [1, 2, 3, 4, 5]
```
- Dictionary (từ điển):
```python
a = {"a": 1, "b": 2, "c": 3}
```
- Đọc và ghi file:
```python
# Đọc file
with open("file.txt", "r") as f:
    data = f.read()
# Ghi file
with open("file.txt", "w") as f:
    f.write("Hello World")
```
- Tham khảo thêm tại: https://www.w3schools.com/python/