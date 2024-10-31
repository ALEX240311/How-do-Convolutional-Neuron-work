#  How do Convolutional Neuron Networks(CNNs) work?

## 1. CNNs (Convolutional Neuron Networks) là gì?

CNN là thuật toán Học Sâu có khả năng tiếp nhận hình ảnh đầu vào, gán các trọng số và độ chệch(Weight & hệ số Bias) cho các đặc trưng hoặc đối tượng khác nhau trong ảnh => từ đó có thể phân biệt chúng với nhau.

Quá trình tiền xử lý trong CNN ít phức tạp hơn so với các thuật toán phân loại khác. Trong khi các phương pháp cổ điển yêu cầu bộ lọc được thiết kế thủ công, CNN có khả năng tự học các bộ lọc đặc trưng với đủ dữ liệu huấn luyện.

CNN thường được sử dụng nhiều trong các bài toán nhận dạng các object trong ảnh. Trong bài viết này ta sẽ tìm hiểu cấu trúc của thuật toán này.


##  2. Tổng quan về CNN
**Trường tiếp nhận cục bộ**: CNN sử dụng các bộ lọc nhỏ được học qua quá trình huấn luyện, di chuyển trên toàn bộ hình ảnh đầu vào, tập trung vào các khu vực cụ thể để phát hiện các đặc trưng cục bộ như cạnh, kết cấu và mẫu hình.

**Chia sẻ trọng số**: Cùng một bộ lọc được áp dụng cho nhiều vùng khác nhau của hình ảnh đầu vào, giúp giảm số lượng tham số và độ phức tạp tính toán. Điều này cho phép mạng có khả năng nhận diện đối tượng dù chúng nằm ở bất kỳ vị trí nào trong hình ảnh.

**Pooling**: Các lớp pooling giảm kích thước không gian của bản đồ đặc trưng, giúp mạng hoạt động hiệu quả hơn và tăng khả năng chống lại các thay đổi nhỏ của dữ liệu đầu vào. Ví dụ, max pooling chọn giá trị lớn nhất trong mỗi vùng, tổng hợp sự hiện diện của đặc trưng trong khi giảm số lượng tham số cần thiết.


## 3. Input Image

Ta đều biết Ảnh là tập hợp của một ma trận pixels, mỗi phần tử của ma trận là đại diện cho độ sáng hoặc màu sắc của một điểm ảnh nhất định trong hình ảnh. Đối với hình ảnh đen trắng(grayscale), mỗi phần tử chỉ chứa một giá trị cường độ(thường là từ 0 -> 255), trong khi đối với hình ảnh màu, mỗi phần tử chứa 3 giá trị riêng biệt đại diện cho ba kênh màu cơ bản đó là Đỏ, Xanh là, và Xanh dương(RGB)

![image](https://hackmd.io/_uploads/BkgwhB80A.png)


*Trên đây chúng ta có 1 hình ảnh RGB đã được tách ra theo 3 kênh màu (Đỏ, Xanh lá, Xanh Dương).*

Khi các hình ảnh có kích thước quá lớn chẳng hạn như 8K(7680x4320), việc xử lý ảnh này sẽ tốn rất nhiều tài nguyên tính toán. Điều này đặt ra thách thức cho các thuật toán xử lý hình ảnh truyền thống.

#### Vai trò của CNN 

Vai trò của CNN chính là giảm kích thước hình ảnh xuống dạng dễ xử lý hơn mà không làm mất đi các đặc trưng quan trọng cần thiết cho việc dự đoán.

CNN sử dụng các lớp tích chập để quét hình ảnh và phát hiện các đặc trưng cục bộ, đồng thời áp dụng kỹ thuật pooling để giảm kích thước mà vẫn giữ lại được thông tin quan trọng


## 4. Cấu trúc của Convolution Neuron Networks
Cấu trúc của CNN, với sự kết hợp giữa trường tiếp nhận cục bộ, chia sẻ trọng số và Pooling, giúp chúng trở nên hiệu quả và đáng tin cậy trong các tác vụ xử lý hình ảnh, phát hiện đối tượng và phân đoạn, biến chúng thành công cụ không thể thiếu trong lĩnh vực thị giác máy tính.
### 4.1 Lớp tích chập(Convolution Layer)
Lớp tích chập là một thành phần chính trong CNN. Như đã đề cập trước đó, hình ảnh đầu vào sẽ được quét qua các lớp tích chập để phát hiện ra các đặc trưng quan trọng. Lớp này hoạt động như một phép toán tích chập bằng cách sử dụng một bộ lọc (hay còn gọi là kernel) trượt qua hình ảnh đầu vào và tính toán tích vô hướng giữa bộ lọc và trường tiếp nhận của đầu vào.

![convoluting-a-5x5x1-image-with-a-3x3x1-kernel-to-get-a-3x3x1-convolved-feature](https://hackmd.io/_uploads/HyKFj88CR.gif)

Ở mô hình này, phần màu xanh lá đại diện cho hình ảnh đầu vào 5x5x1, ký hiệu I. Phần tử tham gia vào phép tích chập trong phần đầu tiên của lớp tích chập được gọi là bộ lọc(Kernel/Filter), ký hiệu là K, được thể hiện bằng màu vàng. Chúng ta đã chọn K là 1 ma trận 3x3x1.

```
Kernel/Filter, K =
1  0  1
0  1  0
1  0  1
```

**Cách Thực Hiện Phép Tích Chập**

**Bước 1 : Đặt Bộ Lọc lên Hình Ảnh:**

``Bộ lọc 3x3 sẽ được đặt lên một phần của hình ảnh 5x5. Phép tích chập sẽ được thực hiện bằng cách quét bộ lọc qua toàn bộ hình ảnh, từng vị trí một.``

**Bước 2 : Tính Toán Tích Chập:**

``Ở mỗi vị trí, nhân từng phần tử của bộ lọc với phần tử tương ứng của hình ảnh và sau đó cộng tất cả các giá trị lại với nhau.
Kết quả sẽ là một giá trị duy nhất cho mỗi vị trí mà bộ lọc được đặt.``



**Bước 3 : Kết quả Đặc trưng:**


``Khi bộ lọc quét qua toàn bộ hình ảnh, nó sẽ tạo ra một bản đồ đặc trưng 
(feature map) có kích thước 3x3x1. 
Bản đồ này chứa các giá trị được tính toán từ phép tích chập.``



Di chuyển của bộ lọc và phép tích chập

Khi thực hiện phép tích chập, bộ lọc sẽ di chuyển qua hình ảnh theo chiều cao và chiều rộng. Ở đây bộ lọc có Stride Length = 1(Không có bước nhảy), điều này có nghĩa là nó sẽ di chuyển từng pixel 1


![image](https://hackmd.io/_uploads/ryGk1P80C.png)


Trong trường hợp bộ lọc sẽ được di chuyển 9 lần bởi vì : 
* Kích thước của hình ảnh đầu vào là 5x5.

* Kích thước của bộ lọc là 3x3.

* Khi bộ lọc được áp dụng vào hình ảnh, nó sẽ tạo ra bản đồ đặc trưng kích thước 3x3.


![image](https://hackmd.io/_uploads/B176Wv8AR.png)


Trong trường hợp này, phép tích chập được thực hiện trên một ma trận hình ảnh có kích thước MxNx3 với bộ lọc 3x3x3. Phép toán được áp dụng trên các kênh màu (RGB), với bộ lọc có độ sâu tương ứng với hình ảnh đầu vào.

Phép nhân ma trận sẽ được thực hiện giữa các kênh của bộ lọc 𝐾 và các kênh tương ứng của hình ảnh đầu vào 𝐼
Sau đó, tất cả các kết quả này sẽ được cộng lại với hệ số bias để tạo ra đầu ra đặc trưng tích chập với độ sâu 1 kênh.

**Same Padding và Valid Padding**
Khi áp dụng các phép tích chập (convolution) trong mạng nơ-ron tích chập (CNN), có hai kỹ thuật padding phổ biến để điều chỉnh kích thước đầu ra:

**Same Padding** 
* Same Padding là kỹ thuật thêm các giá trị padding xung quanh biên của ảnh đầu vào sao cho kích thước đầu ra của phép tích chập giữ nguyên so với kích thước đầu vào. Chẳng hạn, khi áp dụng bộ lọc 3x3x1 lên hình ảnh kích thước 5x5x1, ta sẽ thêm một lớp padding để mở rộng hình ảnh thành 6x6x1. Sau khi thực hiện tích chập, kích thước đầu ra sẽ vẫn là 5x5x1.


* Mục đích của Same Padding là đảm bảo rằng các đặc trưng biên của ảnh không bị mất đi trong quá trình tích chập, giúp mạng học được thông tin từ toàn bộ hình ảnh, bao gồm cả những phần quan trọng ở rìa.

**Valid Padding**
* Valid Padding, ngược lại, không thêm bất kỳ giá trị padding nào vào ảnh đầu vào, khiến cho kích thước đầu ra nhỏ hơn so với đầu vào. Khi áp dụng bộ lọc 3x3x1 lên hình ảnh 5x5x1, kết quả đầu ra sẽ có kích thước 3x3x1.


* Phương pháp này giúp giảm số lượng tính toán cần thiết bằng cách loại bỏ các giá trị biên, nhưng đồng thời cũng có nguy cơ bỏ qua một số thông tin quan trọng ở phần rìa của ảnh.


Tùy thuộc vào yêu cầu bài toán, Same Padding thường được sử dụng khi cần giữ lại toàn bộ thông tin không gian của ảnh, trong khi Valid Padding giúp tối ưu hóa tốc độ tính toán và giảm thiểu số lượng tham số nhưng có thể làm mất thông tin ở vùng biên.


### 4.2 Lớp Pooling(Pooling Layer)

![3x3-pooling-over-5x5-convolved-feature](https://hackmd.io/_uploads/ByYd_OUCR.gif)

Tương tự như lớp Tích chập (Convolutional Layer), lớp Pooling có nhiệm vụ giảm kích thước không gian của bản đồ đặc trưng.

Hiểu đơn giản, lớp Tích chập tạo ra bản đồ đặc trưng từ ảnh đầu vào, và lớp Pooling tiếp tục xử lý bản đồ đặc trưng này để tạo ra một bản đồ đặc trưng mới nhỏ hơn. Mục tiêu của lớp Pooling là giảm thiểu kích thước không gian, giữ lại các đặc trưng quan trọng, từ đó giảm tải tính toán và tăng tính hiệu quả cho mô hình, từ đó tránh tình trạng quá khớp(overfitting).

**Có 2 phép Pooling phổ biến**

1. Max Pooling: Chọn giá trị lớn nhất từ mỗi vùng con của bản đồ đặc trưng. Ví dụ, với kích thước bộ lọc 2x2, max pooling sẽ lấy giá trị lớn nhất từ mỗi vùng 2x2 và bỏ qua các giá trị khác, giúp giữ lại sự hiện diện của đặc trưng mạnh nhất.

2. Average Pooling: Tính trung bình các giá trị trong một vùng con của bản đồ đặc trưng. Phép pooling này thường ít phổ biến hơn max pooling trong các ứng dụng hiện tại.

![image](https://hackmd.io/_uploads/SyCAodICA.png)


Lớp tích chập(Convolution Layer) và lớp Pooling(Pooling Layer) cùng nhau tạo nên lớp thứ i của CNN. Số lượng lớp có thể được tăng lên tùy thuộc vào độ phức tạp của hình ảnh điều này đồng nghĩa với việc tốn nhiều tài nguyên hơn


### 4.3 Làm Phẳng (Flatten)
Sau khi trải qua hai giai đoạn tích chập và pooling, mô hình của chúng ta đã có khả năng hiểu và nhận diện các đặc trưng của hình ảnh. Ở giai đoạn này, việc làm phẳng (Flatten) đầu ra và đưa nó vào mạng nơ-ron truyền thống là cần thiết để thực hiện phân loại.

![flattening](https://hackmd.io/_uploads/HkFpatI00.png)

**Vai trò của Lớp Flatten**
Khi dữ liệu đi qua các lớp tích chập và pooling nó thường được lưu trữ dưới dạng một ma trận 3 chiều(chiều cao, chiều rộng, và số kênh màu).Lớp làm phẳng sẽ chuyển đổi ma trận này thành một vector 1 chiều

Sau khi thực hiện Flattening, tất cả thông tin từ ma trận 3 chiều khi nãy sẽ được "xếp chồng" lại thành 1 chuỗi liên tiếp các giá trị. Điều này giúp đơn giản hóa các bước tiếp theo của mô hình

Việc chuyển đổi sang vector là cần thiết vì các lớp kết nối đầy đủ(Fully Connected Layer) yêu cầu đầu vào là vector để thực hiện các phép toán phân loại hoặc hồi quy. 

Nó cho phép mô hình kết hợp các đặc trưng đã học để dưa ra dự đoán chính xác.
### 4.4 Lớp kết nối đầy đủ(Fully Connected Layers)
Lớp kết nối đầy đủ là 1 thành phần quan trọng trong mạng Neuron đặc biệt trong các mạng CNN. Lớp này chịu trách nhiệm kết nối tất cả các Neuron từ lớp trước đó đến từng Neuron trong lớp hiện tại. Điều này cho phép mô hình học và phát hiện các mối quan hệ phức tạp giữa các đặc trưng đã được trích xuất từ dữ liệu

![image](https://hackmd.io/_uploads/BkQjhqUCC.png)


Hoạt động của lớp kết nối đầy đủ

Đầu vào: Đầu vào cho Lớp Kết Nối Đầy Đủ là một vector đã được làm phẳng từ các lớp trước đó, bao gồm lớp tích chập và lớp pooling. Trong ví dụ này, sau khi trải qua các bước trước, chúng ta có một vector với kích thước 588x1.

Kết nối: Mỗi nút trong lớp kết nối đầy đủ được kết nối với tất cả các nút ở lớp đầu vào. Điều này có nghĩa là mỗi nút sẽ nhận giá trị từ tất cả các đặc trưng đã học từ lớp trước. Mỗi kết nối này đều có một trọng số (weight) và có thể có một độ chệch (bias) đi kèm.

Tính toán đầu ra: Đầu ra của lớp kết nối đầy đủ là một vector mới, được tính toán bằng cách nhân các đầu vào với các trọng số, sau đó cộng thêm độ chệch và áp dụng một hàm kích hoạt như ReLU (Rectified Linear Unit) để tạo ra giá trị đầu ra cho lớp này. Đây là nơi mà mạng có khả năng học các mối quan hệ phi tuyến giữa các đặc trưng.

Phân loại: Kết quả từ lớp kết nối đầy đủ sẽ được đưa vào một lớp đầu ra (Output Layer) với một số nút tương ứng với các lớp mà mô hình dự đoán. Trong trường hợp này, đầu ra sẽ được đưa vào hàm Softmax để tính xác suất cho từng lớp, từ đó giúp phân loại đầu vào vào các danh mục khác nhau.

### 4.5 Dropout
Các lớp Dropout thường được sử dụng trong CNN để ngăn chặn hiện tượng overfitting. Trong quá trình huấn luyện, dropout ngẫu nhiên gán một phần các đơn vị đầu vào về 0 ở mỗi chu kỳ cập nhật. Điều này giúp mô hình trở nên tổng quát hơn bằng cách ngăn nó dựa quá nhiều vào từng neuron riêng lẻ.

### 4.6 Activation function 


## 5. Kết Luận 
Cuối cùng, Mạng Nơ-ron Tích Chập (CNNs) xử lý và phân tích dữ liệu hình ảnh một cách hiệu quả bằng cách sử dụng các lớp tích chập, kích hoạt và pooling. CNNs hoạt động rất tốt trong các nhiệm vụ như phân loại hình ảnh, phát hiện đối tượng và phân đoạn nhờ khả năng tự động học các đặc trưng phân cấp. Khả năng nắm bắt các hệ thống phân cấp không gian đồng thời giảm độ phức tạp tính toán khiến chúng trở nên không thể thiếu trong các ứng dụng thị giác máy tính hiện đại.



*Bài viết là sự đúc kết kiến thức trong quá trình tìm hiểu và học tập vì vậy sẽ có những sai xót nhỏ nhặt. Cảm ơn vì đã dành thời gian để đọc.*
## Sources : 

https://topdev.vn/blog/thuat-toan-cnn-convolutional-neural-network/#cau-truc-mang-cnn

https://www.geeksforgeeks.org/fully-connected-layer-vs-convolutional-layer/

https://saturncloud.io/blog/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way/

https://viblo.asia/p/deep-learning-tim-hieu-ve-mang-tich-chap-cnn-maGK73bOKj2

https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns

