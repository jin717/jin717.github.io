
# 메모장

spark

- spark는 mesos 베이스이다.
- spark는 k8s, DC/OS에 모두 사용가능하다.


진로 고민

- 데이터 사이언스 안에서 어떻게 위치를 잡을 것인가?

NLP

- 자연어 처리는 word embedding이 핵심이다
- word embedding을 하면 아래 내용을 할 수 있다.
    - 유사도/관련도 계산 -> measurement 지정 필요
    - 시각화 -> 시각화 알고리즘 ex) t-sne
    - 벡터 연산
    - 전이 학습
- 임베딩의 의미
    - 말뭉치에 통계량이 있다.
        1. 빈도
        2. 순서
            - 단뱡향 : ELMo 
            - 양방향 : BERT
        3. 분포 : 주변 단어와의 분포 ,맥락
            - word2vec
            - FastText
            - Glove
    - 문장 수준의 임베딩
        - ELMo BERT 동의어 구분 가능
        - ELMo는 GPU가 필요
        - SKT에서 제공하는 BERT pre-trained model 활용 추천
         
    
    
