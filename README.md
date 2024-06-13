# 주택 가격 예측 대회

## 목차
0. 개요
1. 단일 컬럼 분석
2. 결측치 처리 및 이상치 확인
3. 데이터 전처리 및 Feature Engineering
4. 모델 선택 및 최종 결과 확인
5. 향후 개선방향

## 0. 개요
본 보고서는 주택 가격 예측 대회를 위한 데이터 분석 및 모델링 과정을 설명합니다. 데이터 전처리, Feature Engineering, 모델 선택 및 평가, 최종 결과 도출 과정을 상세히 기술하였습니다. 또한 모델 성능 향상을 위한 개선 방안을 제시합니다.

## 1. 단일 컬럼 분석
- 모든 column명을 스네이크 표기법으로 변경
- 범주형과 수치형 데이터로 분류하여 각각 `value_counts()`와 `describe()` 함수 적용
- 특이사항 발견: 조건 컬럼들, 결측 값 모순 등

## 2. 결측치 처리 및 이상치 확인
- 35개 column에서 결측 값 발생
- 명목형 데이터: 결측 값의 최빈값 대체 또는 분석 후 삭제
- 순서형 데이터: 최빈값 또는 상관분석을 통해 대체
- 이산형 및 연속형 데이터: 대부분 결측치를 0으로 대체
- 이상치 처리: 박스 플롯과 산점도를 활용하여 적절히 대체

## 3. 데이터 전처리 및 Feature Engineering
- 명목형 변수: 멀티-핫-인코딩 또는 원-핫-인코딩 적용
- 순서형 변수: 점수 변환 또는 정규화 적용
- 연속형 변수: 추가 변수 생성 및 변환
- 이산형 변수: 새로운 변수 생성 및 변환
- 시계열 변수: 건축 연도와 판매 연도의 차이 등을 이용한 변수 생성

## 4. 모델 선택 및 최종 결과 확인
- 사용한 모델: LightGBM, XGBoost, CatBoost, RandomForest
- 단일 모델 중 CatBoost의 성능이 가장 우수
- 앙상블 적용 결과: 0.12299의 성능 달성 (4843팀 중 374등 달성)

## 5. 향후 개선방향
- Shap value를 통해 중요한 변수들을 이용해 추가적인 Feature Engineering
- 모델의 하이퍼 파라미터 튜닝을 통해 성능 향상 가능
