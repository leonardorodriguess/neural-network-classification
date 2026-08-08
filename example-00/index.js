import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)


    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexa a rede, e mais dados de treino são necessários
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neuronio é positiva, passa para frente
    // Se for zero ou negativa, não passa para frente, e é descartada
    model.add(tf.layers.dense({
        inputShape: [7],
        units: 80, // número de neurônios na camada oculta
        activation: 'relu' // função de ativação ReLU
    }));

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)
    model.add(tf.layers.dense({
        units: 3, // número de neurônios na camada de saída
        activation: 'softmax' // função de ativação softmax para classificação multi-classe
    }));

    // Compilando o modelo
    // optimizer Adam (Adaptive Moment Estimation) é um algoritmo de otimização que ajusta os pesos da rede neural durante o treinamento
    // é um treinamento pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com histórico de errros e acertos

    // loss: categoricalCrossentropy é a função de perda usada para problemas de classificação multi-classe
    // Ele compara o que o modelo acha (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0], medium [0, 1, 0] e basic [0, 0, 1]

    // quanto mais distante da previsão do modelo da responsta correta
    // maior o erro (loss)
    // Exemplo classico de imagens, recomendação categorização de usuários
    // Qualquer coisa em que a resposta certa é "apenas uma entre várias opções" (multi-classe)

    model.compile({
        optimizer: 'adam' ,
        loss: 'categoricalCrossentropy', // função de perda para classificação multi-classe
        metrics: ['accuracy'] // métrica de avaliação do modelo
    });

    // Treinando o modelo
    // verbose: desabilita o log interno (e usa só o callback)
    // epochs: quantas vezes o modelo vai ver os dataset
    // shuffle: embaralha os dados a cada época para evitar overfitting vulgo viés de treino
    await model.fit(inputXs, outputYs, {
        verbose: 0, // 0 = sem log, 1 = log de cada época, 2 = log resumido
        epochs: 100, // número de épocas (quantas vezes o modelo vai ver os dados)
        shuffle: true, // embaralha os dados a cada época para evitar overfitting
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Época ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    })

    return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Criamos o modelo sequencial
// quanto mais dados melhor
// assim o algorimo consegue entender melhor o padrão e fazer previsões mais precisas
const models = trainModel(inputXs, outputYs);   

