# 2021-1 아주대 파란학기(ajou_Paran)


**팀명** : Anything

**도전 과제명** : 가구산업 텍스트, 이미지 복합 검색 시스템 연구 및 개발

**개발 동기** : 가구는 다양한 형태와 패턴이 존재한다. 찾고자 하는 가구가 있을 때 이를 검색하는 것은매우 어렵다. 따라서 자연어 처리 기술(NLP)과, 컴퓨터 비전 기술(CV)을 활용하여 본인이 원하는 형태와 패턴의 가구를 찾을 수 있는 가구 산업 텍스트, 이미지 복합 검색 시스템을 개발한다.

**도전 목표** : 사용자가 텍스트와 이미지를 복합적으로 이용해 특정 패턴이 있는 가구를 검색할 수 있는 검색 엔진 개발

**개요** : 
1. 사용자가 텍스트와 패턴 이미지 입력
2-1. 사용자가 입력한 텍스트는 tensorflow의 Bidirectional GRU를 이용해 학습한 NLP 모델을 통해 22개의 가구 카테고리 중 가장 유사한 가구로 분류
2-2. sklearn.neighbors.KNeighborsClassifier()를 통해 사용자가 입력한 패턴 이미지(GoogleNet 모델을 transfer learning한 CNN 모델로 feature 추출)와 기존에 가지고 있는 4000여장의 패턴 이미지 feature 중 유사한 패턴 이미지 4개 추출
3. yolov5와 cv2.grabCut()을 이용해 2-1에서 분류된 가구의 이미지에서 배경 제거
4. ORB 기술(Oriented FAST and Rotated BRIEF)과 BFMatcher 기술 이용해 "사용자가 입력한 패턴 이미지 + 2-2에서 추출한 4개의 유사 패턴"과 "3에서 배경이 제거된 가구 이미지"에서 다수의 비슷한 벡터가 추출된다면 해당 패턴을 가지고 있는 가구로 판단하여 추출
5. 최종적으로, 4에서 추출한 가구 이미지들을 최종 검색 결과물로 보여줌
6. Django를 이용해 위 과정을 웹으로 구현
7. 웹 호스팅 서비스 Pythonanywhere를 이용해 서버 배포

**22개의 가구 카테고리** :
```
  '가죽소파', '소파베드', 
  '더블/퀸/킹침대', '싱글/수퍼싱글+침대', '디반침대', 
  '옷장', '어린이옷장', 
  '바테이블', '커피테이블/보조테이블', '식탁',  
  '식탁의자', '바의자', '사무용의자', '스툴/벤치', '어린이의자', '영아용의자', '가죽암체어', '라탄암체어', '리클라이너', 
  '커튼', '샤워커튼', '블라인드'
```

---

**Work Flow**

![1](https://user-images.githubusercontent.com/62659407/121677965-87404f80-caf1-11eb-9c23-2248f7cc5166.png)

---

**Prototype**

- "화장실에 쓰는 커튼" + 특정 패턴 이미지 검색 → 특정 패턴 이미지가 있는 "샤워커튼" 출력

<img src="https://user-images.githubusercontent.com/62659407/121678434-25341a00-caf2-11eb-84c1-3e4767bd41aa.png" width="70%">
<img src="https://user-images.githubusercontent.com/62659407/121678565-498ff680-caf2-11eb-8458-e73701109e53.png" width="60%">


- "야자로 만든 의자" + 특정 패턴 이미지 검색 → 특정 패턴 이미지가 있는 "라탄 암체어" 출력

<img src="https://user-images.githubusercontent.com/62659407/121678603-544a8b80-caf2-11eb-8d55-1f527b27dbf6.png" width="60%">
<img src="https://user-images.githubusercontent.com/62659407/121678636-60364d80-caf2-11eb-9ca3-ab4148c00be7.png" width="60%">


- "가죽으로 만든 소파" + 특정 패턴 이미지 검색 → 특정 패턴 이미지가 있는 "가죽소파" 출력


<img src="https://user-images.githubusercontent.com/62659407/121678684-70e6c380-caf2-11eb-9c9e-e6775e3b4a7b.png" width="60%">
<img src="https://user-images.githubusercontent.com/62659407/121678656-662c2e80-caf2-11eb-9ba4-34d68d883910.png" width="60%">

- "매트리스 침대" + 특정 패턴 이미지 검색 → 특정 패턴 이미지가 있는 "디반침대" 출력


<img src="https://user-images.githubusercontent.com/62659407/121678746-865bed80-caf2-11eb-9cf8-f1104e1885e7.png" width="60%">
<img src="https://user-images.githubusercontent.com/62659407/121678771-8cea6500-caf2-11eb-98f8-36f301bd015f.png" width="60%">


---

**이노베이터 상 수상**
<br><br>
<img src="https://user-images.githubusercontent.com/62659407/140987773-c4f7fec6-9606-4390-844f-45a051b47c45.png" width="40%">
<br><br>
**가출원 등록**
<br><br>
<img src="https://user-images.githubusercontent.com/62659407/121677988-8efff400-caf1-11eb-8137-239e3789af4b.png" width="40%">
<br>

---
   
## NLP를 이용해 가구 분류 모델 구현(NLP_BiGRU)

- 개요
  + 22종의 가구를 분류하는 인공지능 모델
  + 학습 데이터는 IKEA에서 웹 크롤링한 텍스트 데이터
  + Bidirectional GRU 알고리즘 이용

- 구현 사항
  + ikea_crawling.ipynb() : BeautifulSoup 라이브러리를 이용해 학습데이터로 사용될 텍스트 데이터를 IKEA의 [가구 설명]을 웹 크롤링<br><br>
  + NLP_BiGRU.ipynb() :
  + konlpy 라이브러이의 okt() 형태소 분석기를 이용해 텍스트 데이터 토큰화
  + 단어의 빈도수를 이용해 텍스트 데이터 정수 인코딩
  + 텍스트 데이터 중 최장 길이로 패딩
  + 위 과정을 거쳐 전처리된 텍스트 데이터를 Embedding()을 이용해 단어 임베딩
  + Tensorflow 라이브러리의 Bidirectional GRU 알고리즘을 이용해 인공지능 모델 학습 및 검증
  + 구현한 NLP 모델을 'anythingNLP.h5'로 저장

- 검증 정확도
```
  ...중략...
  Epoch 00020: val_acc did not improve from 0.99342
```

---

## 사용자가 입력한 패턴이 있는 가구 이미지 추출 모델 구현(CV_furniture)

- 개요
  + 입력 데이터는 Google에서 웹 크롤링한 가구 이미지(CV_furniture/images/) 중 NLP에서 추출된 가구의 이미지
  + 사용자가 입력한 패턴 + 유사 패턴 4개 추출 → 총 5개의 패턴으로 가구 이미지와 벡터 비교
  + 일정 기준 이상의 벡터가 발견되면 사용자가 입력한 패턴이 있는 가구로 판단해 최종 결과물로 출력

- 구현 사항
  + image_webcrawling.ipynb : selenium 라이브러리와 Google 드라이버를 이용해 입력 데이터인 가구 이미지를 Google에서 웹 크롤링<br><br>
  + CV.ipynb : 
  + sklearn.neighbors.KNeighborsClassifier()를 통해 사용자가 입력한 패턴 이미지(GoogleNet 모델을 transfer learning한 CNN 모델로 feature 추출)와 기존에 가지고 있는 4000여장의 패턴 이미지(CV_furniture/DTD/) feature 중 유사한 패턴 이미지 4개 추출 → 총 5개의 패턴 사용
  + yolov5와 cv2.grabCut()을 이용해 입력 데이터의 가구 이미지에서 가구 인식의 효율을 높이기 위해 배경 제거
  + ORB 기술(Oriented FAST and Rotated BRIEF)과 BFMatcher 기술을 이용해 위에서 구한 5개의 패턴과 배경이 제거된 가구 이미지 벡터 비교 → 다수의 비슷한 벡터가 추출된다면 해당 패턴을 가지고 있는 가구로 판단하여 최종 결과물로 출력
  + 

- GoogleNet 모델 정확도

<img src = "https://user-images.githubusercontent.com/62659407/121727147-7bbb4b80-cb26-11eb-878c-c6a35fa9b997.png" width="30%">

```
  76%
```

- 사용자가 입력한 패턴과 유사 패턴 유사도

<img src="https://user-images.githubusercontent.com/62659407/121727175-870e7700-cb26-11eb-9791-9415fcc5564e.png" width="30%">

```
  83.57%
```

---

## Django를 이용해 프론트엔드 + 백엔드(NLP와 CV 모델 결합) 웹 구현(anything_site)

- 개요
  + Django를 이용해 위 과정을 결합해 백엔드 구현
  + XD를 이용해 프론트엔드 구현
  + 사용자가 텍스트와 패턴 이미지를 검색했을 때, 패턴 이미지가 있는 해당 가구의 이미지가 로드되는 웹 구현

- 구현 사항
  + anything_site/form/templates/main_page.html : HTML에서 form 기능을 이용해 사용자가 텍스트 및 패턴 이미지 업로드<br><br>
  + anything_site/form/views.py : 
  + request.POST를 이용해 전달받은 사용자가 입력한 텍스트를 NLP 모델을 통해 22개의 가구 카테고리 중 가장 유사한 특정 가구로 추출
  + request.FILES를 이용해 전달받은 사용자가 입력한 패턴 이미지를 CV 모델을 통해 유사 패턴 이미지 추출 및 해당 패턴이 있는 가구 이미지 추출<br><br>
  + anything_site/form/templates/searched_page.html : 
  + 사용자가 입력한 텍스트 및 NLP 모델을 통해 추출된 가구 카테고리 로드
  + 사용자가 입력한 패턴 이미지 및 CV 모델을 통해 추출된 유사 패턴 이미지 4개 로드
  + 해당 패턴이 있는 가구 이미지 로드 <br><br>
  + anything_site/static/style : XD를 이용해 프론트엔드 구현(.HTML, .CSS, .JS)<br><br>
  + 참고 사항 :
  + 패턴 이미지 4000여장(anything_site/static/DTD), 가구 이미지(anything_site/static/images), 프론트엔드 파일(anything_site/static/style) → Django의 static 파일 구조로 관리  
  + 사용자가 업로드하는 패턴 이미지 → anything_site/media에 동적으로 저장 → Django의 media 파일 구조로 관리 

---
