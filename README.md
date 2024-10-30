# Machine Learning(AI) 

Turma da Manha (A)

Guilherme de Sousa Cirumbolo - 00330049
Pedro Marhofer Alles - 00326188

## Linear Regression - alegrete.ipynb

Valores iniciais: Utilizamos os numeros de cartao UFRGS.

B = 3  
W = 3  
Valores de alpha = 0.0049  
Num_iterations = 3261880  

Melhor erro quadrático médio: 8.5277081  
Curva encontrada: 1.1606254547625439*x + -3.4500333927330717

Conclusao:  
Com base nos resultados, o modelo encontrou uma relacao linear entre a area do terreno e o preco, onde o coeficiente angular (w = 1.16), que confirma uma correlacao positiva entre area e preco, e coeficiente linear (b = -3.45).  
A taxa de aprendizado escolhida foi alpha = 0.0049. Esse valor baixo foi necessario para garantir a estabilidade da descida de gradiente e evitar que o modelo "explodisse" (como aconteceu com o valor inicial de alpha = 0.1).
Apesar de usarmos 3261880 iteracoes, com 10000 ja obtivemos um erro quadratico medio similar.  
Dessa forma, apesar do grande numero de iteraoes, o modelo está bem ajustado.

## Neural Network - Trabalho_redes_neurais.ipynb

Análise dos datasets:
- CIFAR-10->   Classes: 10 | Amostras: 60.000 | Tamanho: 32x32 pixels | Canais de cor: 3
- CIFAR-100-> Classes: 100 | Amostras: 60.000 | Tamanho: 32x32 pixels | Canais de cor: 3
- MNIST-> Classes: 10 | Amostras: 70.000 | Tamanho: 28x28 pixels | Canais de cor: 1
- Fashion MNIST-> Classes: 10 | Amostras: 70.000 | Tamanho: 28x28 pixels | Canais de cor: 1

1) Ranking de complexidade:     
- MNIST:        
 É o mais simples pelas imagens serem em escala de cinza, baixa variedade entre as classes por serem digitos padronizados, e resolucao baixa(28x28)      
- FMNIST:       
Possui as mesmas simplicidades de cor cinza e resolucao baixa, mas é mais dificil que MNIST por seu conjunto de classes com diferentes tipos de roupas e acessórios que podem ter formas e texturas mais complexas.       
- CIFAR-10:         
É mais difícil que os anteriores, por conta de possuir maior resolucao (32x32), possui 3 camadas de cores formando RGB, diferenca claras entre as classes (passaros, carros, cavalos), e ambiguidade de padroes de fundo, onde um passaro e um aviao possam parecer semelhantes, dificultando o aprendizado  
- CIFAR-100:  
É o mais difícil entre os quatro datasets devido ao aumento consideravel de classes(100 classes), possuir 3 canais de cor formando o RGB, e possuir classes proximas semanticamente (diferentes tipos de animais e/ou veiculos) o que exige da maquina um aprendizado grande para reconhecer diferenças pequenas.  


2) Mudancas das configuracoes dos datasets:
- 1: Mudanca de otimizadores(adam, Nadam, SGD, SGD com momentum, RMSprop)  
Apesar da melhoria em velocidade de execucao do SGD e RMSprop, a melhor precisao foi do adam e Nadam. 
- 2: Tamanho das Camadas (Rede Rasa(tanh) x Rede Profunda(relu))   
O tamanho das camadas influenciam diretamente a relação entre tempo de treinamento e precisão. Dessa forma, a rede profunda com ativador relu, apesar de mais lenta, foi a com maior precisao nos testes.
- 3: Quantidade de Neuronios  
Mais neurônios em uma camada aumentam a capacidade de aprendizado da rede, melhorando a precisão, mas com custos de tempo de treinamento.
- 4: Taxa de Dropout  
Dropout melhora a robustez e a capacidade de generalização do modelo, o que pode indiretamente resultar em melhor precisão no conjunto de teste, mas não necessariamente garante melhor precisão.
- 5: Tamanho do Batch  
Batches menores podem ajudar a melhorar a precisão, enquanto batches maiores podem acelerar o treinamento.

Conclusao:      
Os quatro datasets (MNIST, Fashion MNIST, CIFAR-10 e CIFAR-100) representam uma escala de complexidade crescente, onde elementos como o número de camadas, otimizadores, dimensão das imagens afetam diretamente a dificuldade de cada um. Dessa forma, com o aumento da complexidade visual e do número de categorias, esses datasets exigem modelos mais robustos e arquiteturas mais profundas para capturar as nuances e diferenças entre as classes.

## Documentação:
- Python 3
- Anaconda
- IPython Kernel
- TensorFlow
- Import time
