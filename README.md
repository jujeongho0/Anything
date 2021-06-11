# 2021-1 아주대 파란학기(ajou_Paran)


**팀명** : Anything

**도전 과제명** : 가구산업 텍스트, 이미지 복합 검색 시스템 연구 및 개발

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

---

## Django를 이용해 프론트엔드 + 백엔드(NLP와 CV 모델 결합) 웹 구현(anything_site)

- 개요
  + Django를 이용해 위 과정을 결합해 백엔드 구현
  + XD를 이용해 프론트엔드 구현
  + 사용자가 텍스트와 패턴 이미지를 검색했을 때, 패턴 이미지가 있는 해당 가구의 이미지가 로드되는 웹 구현

- 구현 사항
  + anything_site/form/templates/main_page.html : HTML에서 form 기능을 이용해 사용자가 텍스트 및 이미지 업로드<br><br>
  + CV.ipynb : 
  + sklearn.neighbors.KNeighborsClassifier()를 통해 사용자가 입력한 패턴 이미지(GoogleNet 모델을 transfer learning한 CNN 모델로 feature 추출)와 기존에 가지고 있는 4000여장의 패턴 이미지(CV_furniture/DTD/) feature 중 유사한 패턴 이미지 4개 추출 → 총 5개의 패턴 사용
  + yolov5와 cv2.grabCut()을 이용해 입력 데이터의 가구 이미지에서 가구 인식의 효율을 높이기 위해 배경 제거
  + ORB 기술(Oriented FAST and Rotated BRIEF)과 BFMatcher 기술을 이용해 위에서 구한 5개의 패턴과 배경이 제거된 가구 이미지 벡터 비교 → 다수의 비슷한 벡터가 추출된다면 해당 패턴을 가지고 있는 가구로 판단하여 최종 결과물로 출력

---
