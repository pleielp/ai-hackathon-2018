# 수정 내역
## 코드 해설 on [/missions/examples/kin/study](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/)
* [main.py](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/main.py)
  - from dataset import KinQueryDataset, preprocess
  - argument들을 config에 할당.
  - 변수 및 placeholder 설정.
    - x = tf.placeholder(tf.float32, [None, hidden_layer_size])
    - y = tf.placeholder(tf.float32, [None, output_size])
    - w = tf.Variable(tf.random_normal([hidden_layer_size, output_size]), name="w")
    - b = tf.Variable(tf.random_normal([output_size]), name="b")
  - 변수 계산.
    - hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
    - cost = -(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
    - train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    - prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    - accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))
  - mode 'train'
    - 데이터를 불러와서 train.
  - mode 'test_debug'
    - 저장된 .ckpt와 데이터를 불러와서 test.
  - mode 'test_local'
    - for NSML
* [dataset.py](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/dataset.py):
  - from char_parser import vectorize_str
  - class KinQueryDataset
    - train_data 파일을 열어 f.readlines() 메소드를 preprocess()에 data 인수로 입력해 나온 출력을 self.queries에 할당.
    - train_label 파일을 열고 (data_length, 1) shape np.array에 넣어 self.labels에 할당.
  - preprocess(data: list, max_length: int, hidden_layer_size: int)
    - splitBytab()으로 질문쌍을 두 질문 list로 분리
    - vectorize_str()로 벡터화
    - padding()으로 max_length까지 zero padding
    - BasicLSTMCell(hidden_layer_size) 선언 후 dynamic_rnn()
    - 두 질문의 dynamic_rnn() 출력 맨 마지막 값을 비교해 유사도 계산
    - return (data_length, 1(유사도))) shape np.array.
* [char_parser.py](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/char_parser.py)
  - vectorize_chr(char)
    - 문자 하나를 인수로 받아 한글(초성/중성/종성), 영어, 숫자, 특수문자에 따라 벡터화.
    - return vector
  - vectorize_str(str): 
    - 문자열을 인수로 받아 vectorize_chr()에 문자 하나씩 입력.
    - return (string_length, vector_size(default 6)) shape 리스트.
* [setup.py](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/setup.py)
  - NSML 라이브러리 setup
* [/save](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/save/), [/save2](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/save2/)
  - test_main.py에서 학습한 session을 저장한 .ckpt
  - /save: learning_rate = 0.001
  - /save2: learning_rate = 0.0001
* [/test](https://github.com/pleielp/ai-hackathon-2018/blob/master/missions/examples/kin/study/test)
  - test, debug, backup 파일들

## 해결해야 할 것
* 학습이 무의미
  - tf.nn.dynamic_rnn()이 매번 다른 outputs를 출력 -> dataset이 매번 다르게 로딩됨.
  - train 할 때 한 dataset에 너무 오버피팅.
  - 따라서 학습된 session 불러와 똑같은 데이터에 test_debug해도 다르게 로딩된 dataset에는 재학습을 해야한다.

## 참고
- [모두를 위한 딥러닝 강좌 시즌 1](https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=1_)
- [TensorFlow API](https://www.tensorflow.org/api_docs/python/tf)
- [TensorFlow 한글 문서](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/)

- [TF의 텐서와 상수, 변수, 플레이스홀더](https://tensorflow.blog/2017/05/10/tf%EC%9D%98-%ED%85%90%EC%84%9C%EC%99%80-%EC%83%81%EC%88%98-%EB%B3%80%EC%88%98-%ED%94%8C%EB%A0%88%EC%9D%B4%EC%8A%A4%ED%99%80%EB%8D%94/)
- [RNN Tutorial Part 4 - GRU/LSTM RNN 구조를 Python과 Theano를 이용하여 구현하기](http://aikorea.org/blog/rnn-tutorial-4/)
- [Gradient Descent Optimization Algorithms 정리](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)
- [word2vec_basic.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
- [word2vec.py](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)
- [One-hot 인코딩 쉽게 하기](https://minjejeon.github.io/learningstock/2017/06/05/easy-one-hot-encoding.html)
- [인공지능을 위한 선형대수](http://www.edwith.org/linearalgebra4ai/lecture/22720/)

- [List of lists into numpy array](https://stackoverflow.com/questions/10346336/list-of-lists-into-numpy-array/26224619)

<br>

### 이하는 NAVER AI HACKATHON 공지사항
---

![banner](./res/NSMLHack_web_1000x260_G.jpg)

# 공지사항
### 미션 공개
[미션](#%EB%AF%B8%EC%85%98)을 공개합니다.<br>
[NSML](https://hack.nsml.navercorp.com/download)을 다시 다운로드하고 설치해 주세요!

예선 라운드 일정이 변경됐습니다.
* 예선 1라운드
  - 기존 2018년 4월 2일(월) ~ 2018년 4월 8일(일)
  - 변경 2018년 4월 2일(월) ~ 2018년 4월 9일(월) 오전 11시
* 예선 2라운드
  - 기존 2018년 4월 9일(월) ~ 2018년 4월 15일(일)
  - 기존 2018년 4월 10일(화) 오전 11시 ~ 2018년 4월 16일(월) 오전 11시

# 네이버 AI 해커톤 2018

"한계를 넘어 상상에 도전하자!"

인간이 오감을 활용하는 것처럼 AI도 인간의 오감을 모두 활용하는 방향으로 나아갈 것입니다.<br>
또한, 인터넷과 모바일이 세상을 크게 변화시킨 것처럼 AI 역시 세상을 크게 변화시킬 것이며 그 영향력은 더욱 커질 것입니다.<br>
네이버는 AI와 함께 더 편리하고 행복한 미래를 만들기 위해 **네이버 AI 해커톤 2018**을 준비했습니다.<br>

특히, 이번 네이버 AI 해커톤 2018은 네이버의 클라우드 머신러닝 플랫폼인 <strong>[NSML](https://hack.nsml.navercorp.com/intro)</strong>과 함께 합니다.

<strong>NSML(Naver Smart Machine Learning)</strong>은 모델을 연구하고 개발하는 데 필요한 복잡한 과정을 대신 처리해주어<br>
연구 개발자들이 "모델 개발"에만 전념할 수 있고, 다양한 시도를 쉽게 할 수 있는 창의적인 환경을 제공할 것입니다.

AI를 통해 복잡한 문제를 해결하고 싶나요?<br>
AI 전문가들과 함께 문제 해결 방법을 고민하고 경험을 공유하고 싶다고요?

지금 바로 <strong>네이버 AI 해커톤 2018</strong>에 참여해서<br>
서로의 경험을 공유하고, 다양하고 창의적인 방법으로 문제를 해결해 보세요!

[![안내 및 문제 소개](res/cSGPHtzPFQ.png)](https://youtu.be/cSGPHtzPFQw)

## 멘토
여러분들과 함께 문제 해결 방법을 고민하고 조언 해주실 슈퍼 멘토를 소개합니다.

<table>
  <tr style="background-color:#fff">
    <td style="text-align:center">
      <img src="res/ksh.jpg" width="100"><br>
      김성훈
    </td>
    <td style="text-align:center">
      <img src="res/kdh.jpg"><br>
      곽동현
    </td>
    <td style="text-align:center">
      <img src="res/smj.jpg"><br>
      서민준
    </td>
    <td style="text-align:center">
      <img src="res/shj.jpg"><br>
      송현제
    </td>
    <td style="text-align:center">
      <img src="res/ckh.jpg"><br>
      최경호
    </td>
  </tr>
</table>

## 참가 신청
AI로 문제를 해결하는 데 관심 있는 분이라면 누구나 참가 신청할 수 있습니다.<br>
<strong>개인 또는 팀(최대 3명)</strong>으로 참가 가능합니다. [네이버 폼](http://naver.me/GyfLHzwg)으로 참가 신청하세요!

* **신청기간**: 2018년 3월 12일(월)~3월 25일(일)
* **참가 신청 폼**: 참가 신청 마감
* 신청자가 많을 경우 심사 후 개별 안내

## 일정
<table class="tbl_schedule">
  <tr>
    <th style="text-align:left;width:50%">일정</th>
    <th style="text-align:center;width:15%">기간</th>
    <th style="text-align:left;width:35%">장소</th>
  </tr>
  <tr>
    <td>
      <strong>참가 신청</strong><br>
      2018년 3월 12일(월)~3월 25일(일)
    </td>
    <td style="text-align:center">2주</td>
    <td>
      참가 신청 마감
    </td>
  </tr>
  <tr>
    <td>
      <strong>예선</strong><br>
      2018년 4월 2일(월)~4월 16일(월)
    </td>
    <td style="text-align:center">2주</td>
    <td>
      온라인<br>
      <a href="https://hack.nsml.navercorp.com">https://hack.nsml.navercorp.com</a>
    </td>
  </tr>
  <tr>
    <td>
      <strong>결선</strong><br>
      2018년 4월 26일(목)~4월 27일(금)
    </td>
    <td style="text-align:center">1박 2일</td>
    <td>
      네이버 커넥트원(춘천)<br>
    </td>
  </tr>
</table>

> ※ 예선 및 결선 참가자에게는 개별로 참가 안내드립니다.<br>
> &nbsp;&nbsp;&nbsp;결선 참가자는 네이버 본사(그린팩토리, 분당)에 모여서 커넥트원(춘천)으로 함께 이동하며<br>
&nbsp;&nbsp;&nbsp;네이버 본사 - 커넥트원 간 이동 차량 및 결선 기간 중 숙식, 간식 등을 제공합니다.

## 미션
* [네이버 지식iN 질문 유사도 예측](missions/kin.md)
* [네이버 영화 평점 예측](missions/movie-review.md)

> ※ 모든 미션은 NSML 플랫폼을 사용해 해결합니다.<br>
> &nbsp;&nbsp;&nbsp;NSML을 통해 미션을 해결하는 방법은 이 [튜토리얼](missions/tutorial.md)을 참고해 주세요.

## 진행 방식 및 심사 기준

### 예선

* 예선 참가자에게는 예선 기간 중 매일 오전 11시에 600 NSML 크레딧을 지급합니다.
* 팀 참가자일 경우 대표 팀원에게만 지급합니다.
* 사용하지 않는 크레딧은 누적됩니다.

#### ***예선 1라운드***
* 2018년 4월 2일(월) ~ 2018년 4월 9일(월) 오전 11시
* NSML 리더보드 순위로 2라운드 진출자 선정. 순위가 낮으면 자동 컷오프.

#### ***예선 2라운드***
* 2018년 4월 10일(화) 오전 11시 ~ 2018년 4월 16일(월) 오전 11시
* NSML 리더보드 순위로 결선 진출자 선정

### 결선
* 2018년 4월 26일(목) ~ 4월 27일(금) 1박 2일 동안 진행
* 결선 참가자에게는 600 + α NSML 크레딧을 지급합니다.
* NSML 리더보드 순위로 최종 순위를 결정합니다.

> ※ 1 NSML 크레딧으로 NSML GPU를 1분 사용할 수 있습니다.<br>
> &nbsp;&nbsp;&nbsp;10 NSML 크레딧 = GPU 1개 * 10분 = GPU 2개 * 5분 사용

> ※ 예선, 결선 진출자는 개별 안내 드립니다.


## 시상 및 혜택

* 총 1000만 원 상당의 상금(각 미션별 시상) 및 기념품
* 총 1억 원 상당의 [네이버 클라우드 플랫폼 크레딧](FAQ.md#q-%EB%84%A4%EC%9D%B4%EB%B2%84-%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C-%ED%94%8C%EB%9E%AB%ED%8F%BC-%ED%81%AC%EB%A0%88%EB%94%A7%EC%9D%80-%EB%AD%94%EA%B0%80%EC%9A%94) 지급
* 결선 진출자에게는 티셔츠 등의 기념품 증정
* 우수 참가자 중 네이버 인턴 지원 시 서류 전형 면제

## FAQ
자무 문의하는 내용을 확인해 보세요! [FAQ.md](FAQ.md)

## 문의
해커톤 관련 문의는 아래 이메일을 통해 할 수 있습니다.<br>
dl_ai_hackathon_2018@navercorp.com

## License
```
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
