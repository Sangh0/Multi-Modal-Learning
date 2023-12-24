# Learning Transferable Visual Models From Natural Lanuage Supervision  
- paper review  

## Abstract  
- SOTA vision model은 fixed dataset을 가지고 학습
- 그러나 이는 generality와 usability를 제한  
- 그래서 이미지에 대한 raw text를 가지고 학습하는 method 즉, CLIP을 제안  
- CLIP은 ImageNet에서 zero-shot transfer를 수행한다면 original ResNet50의 성능과 비슷

## Introduction and Motivating Work  
- NLP에서 raw text에 관한 연구가 지속되어 오고 있음  
- high quality labeled dataset보다 web-scale dataset이 더 우수한 supervision을 제공
- 그러나 Vision 분야에서는 여전히 ImageNet과 같은 crowd-labeled dataset을 사용
- Vision 분야에서도 NLP에서 web-scale dataset을 이용한 것처럼 scalable pre-training method를 적용할 수 있지 않을까?
- 그래서 이전 연구들을 찾아보자면 image representation을 위한 NLP supervision에 대한 연구가 있는데 굉장히 레어함
- 심지어 성능은 ImageNet 기준으로 11.5%의 정확도일 정도로 굉장히 좋지 않음  
- 그러나 weakly supervised method에서는 약간 성능 향상이 있었음
- 이 둘의 결정적 차이는 scale임
- 그래서 이 논문에서는 400M개의 image-text pari를 수집했고 
- ConVIRT의 simplified 버전을 이용해 학습했으며 이를 CLIP이라 명명

## Approach  
### Natural Language Supervision 
- natural language로 학습하는 것은 다른 training method보다 잠재성이 있음  
- annotation이 필요한 다른 분야에 비해 scaling 조절이 쉽고 annotation이 필요없다는 것이 가장 큰 강점
- 또한 unsupervised, self-supervised 방식보다 더 중요한 장점을 갖고 있는데
- 단순히 representation 뿐만 아니라 language와 견결해 flexible하게 zero-shot transfer까지 가능

### Selecting an Efficient Pre-Training Method

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure1.png?raw=true" width=600>

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure2.png?raw=true" width=600>

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure3.png?raw=true" width=600>

- SOTA Vision system들은 많은 연산량을 필요로 함
- 그래서 효율적으로 학습할 방법을 고민
- initial approach로 VirTex와 비슷하며 image caption을 예측하기 위해 image CNN과 text transformer를 동시에 훈련
- 그러나 이 방법은 모델을 scaling하기가 어렵다는 단점 존재
- 그래서 contrastive learning method로 captioning task를 수행하도록 수정
- 위의 figure 2에 나와있듯이 효율적 학습이 가능함
- 이때 문장 내에서 단어의 순서는 고려하지 않기로 결정
- generative method도 있지만 이는 동일한 성능을 내는 contrastive method에 비해 필요한 computing 규모가 더 커서 비효율적
- 이러한 점들을 고려해 더 쉬운 proxy task를 해결하도록 탐색
- 즉, 정확한 단어가 아닌 어떤 텍스트 전체가 어떤 이미지와 연결되어 있는지
- 따라서 CLIP은 image와 text embedding의 $N$개의 pair에 대한 cosine similarity를 최대화하고 $N^2-N$개 embedding의 similarity는 최소화
- 이를 위해 image와 text encoder를 같이 학습함으로써 multi-modal space를 학습
- CLIP의 pipeline 또는 architecture는 위의 figure 1을, pseudo code는 figure 3를 참고
- CLIP을 학습할 때 image encoder와 text encoder 둘 다 pre-trained weight로 initialization하지 않음
- 이에 대한 이유로는 데이터셋의 규모가 굉장히 크기 때문에
- 그리고 각 encoder의 representation을 multi-modal space로 mapping할 때 linear projection을 사용
- data augmentation으로는 random square crop from resized images만을 사용
- softmax range를 flexible하게 만들어주는 temperature paramter $\tau$는 학습하면서 log-parameterized로 설정하며 이는 hyperparameter로서의 기능을 피하기 위함

### Choosing and Scaling a Model
- image encoder로는 크게 2개의 architecture를 채택
- 먼저, ResNet-50을 base로 사용하고 여기에 modification을 추가해 ResNet-D 등을 채택
- 두 번째로는, ViT를 채택
- text encoder로는 Transformer를 채택하며 Masked self-attention이 사용됨

### Training
- 5개 버전의 ResNet, 3개 버전의 ViT를 사용
- ResNet-50, 101, 그리고 EfficientNet-style model scaling 메소드를 적용해 4x, 16x, 64x 버전의 모델 채택
- ViT의 경우, ViT-B/32, ViT-B/16, ViT-L/14를 채택
- 32 epochs, Adam Optimizer with deoupled weight decay regularization 즉, AdamW를 사용함
- learning rate scheduling을 위해 cosine scheduler 사용
- initial hyperparamter는 grid search, random search, manual tuning 등의 작업을 통해 세팅
- learnable temperature paramter $\tau$는 initial value로 0.07로 설정
- minibatch size는 32768

## Experiments  
### Zero-Shot Transfer  
#### Using CLIP For Zero-Shor Transfer  
- 먼저, image의 feature embedding과 이에 해당하는 pair의 text feature embedding을 계산
- 이후 cosine similarity을 이용해 두 embedding vector를 연산
- 이때 temperature parameter $\tau$가 쓰이고 $\tau$에 의해 normalized 됨
- prediction layer는 multinomial logistic regression classifier이며 L2-normalized inputs, L2-normalized weights, no bias, temperature scaling이 활용됨

#### Initial Comparison to Visual N-Grams  

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/table1.png?raw=true" width=600>

- CLIP이 Visual N-Grams보다 월등히 성능이 좋은 것을 알 수 있음  

#### Prompt Engineering and Ensembling  

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure4.png?raw=true" width=600>

- 클래스의 이름이 CLIP의 text encoder에 제공되는 유일한 정보일 때 context의 부재로 인해 그 의미를 정확히 이해할 수 없음  
- 그래서 본 논문의 저자들은 prompt template 즉, "A photo of a {label}"을 사용함으로써 이를 어느 정도 해결  
- 이로 인해 ImageNet 기준으로 1.3%의 improvement가 발생
- 또한 클래스의 카테고리를 추가한 것도 꽤 큰 도움이 됐었음
- 예를 들자면, "A photo of a {label}, a type of pet"
- 따라서 이러한 prompt들을 ensemble하여 ImageNet 기준으로 단일 prompt보다 성능을 3.5% 개선

#### Analysis of Zero-Shot CLIP Performance

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure5.png?raw=true" width=600>

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure6.png?raw=true" width=600>

- figure 5에서 총 27개의 benchmark 중, 16개에서 CLIP이 더 좋은 성능을 보여줌
- CLIP이 낮은 성능을 보인 benchmark는 주로 복잡하거나 추상적인 dataset임  

- figure 6을 보면 zero-shot CLIP을 few-shot 모델들과 비교했는데 그 결과를 나타냈다
- BiT-M 16 shot learning이랑 zero-shot CLIP의 성능이 비슷하다

#### Representation Learning  

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure10.png?raw=true" width=600>

- CLIP이 모든 method 또는 model들보다 성능이 우수하다  

#### Robustness to Natural Distribution Shift  

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/figure12.png?raw=true" width=600>

- 이전에 연구되었던 내용들에 의하면 사람보다 성능이 좋다고 주장을 했지만 사실 training data의 distribution을 벗어나는 case에 대해서는 robust하지 못한 성능을 보여준다  
- 이런 점에서, 직관적으로 바라봐서, zero-shot은 논리적으로 data의 패턴을 학습한다  
- 따라서 CLIP은 기존 method들보다 우수한 성능을 보여주며 위의 figure 12에 나타나있다


## Comparison to Human Performance

<img src = "https://github.com/Sangh0/Multi-Modal-Learning/blob/main/CLIP/figures/table2.png?raw=true" width=600>

- 이 논문에서 굉장히 흥미로운 실험들을 했는데 그 중 하나가 사람에 대한 성능이다 
- 인간의 성능을 측정하기 위해 5명을 대상으로 Oxford IIT Pets dataset을 분류하도록 실험 
- 3669개의 이미지를 보고 37개의 cat or dog의 품종을 분류하도록 수행
- 위의 table 2에 실험 결과가 나타나 있음
- 흥미롭게도 인간은 클래스당 하나의 학습 샘플만으로도 평균 정확도를 54%에서 76%로 끌어올림
- 그리고 학습 샘플을 추가할수록 이 gap은 줄어듦
- 즉, 인간은 본인이 무엇을 모르는지 알 수 있으며 단 하나의 샘플로 이 능력을 업데이트할 수 있음
- 이러한 관점에서 본다면 CLIP은 zero-shot에서 유망한 method라고 볼 수 있음

## Limitations  
- CLIP의 task learning과 transfer capabilities을 개선시키기 위한 노력은 아직 필요
- 또한 성능도 fine-grained classification에서는 부족
- 그리고 out-of-distribution의 data에 대해서는 generalized한 성능을 보여주지 못함
- CLIP은 internet에 존재하는 image와 text pair을 수집해 학습함
- 이러한 pair들은 filtering되지 않아 어느 정도의 social bias가 들어있음
- 그럼에도 불구하고 이 논문은 Vision 분야에서 task-agnostic의 새로운 objective를 제안한 논문이며 활용도가 굉장히 높은 method라고 생각