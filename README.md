PharmaNexus: Market Intelligence & Forecast System

🚀 Visão Geral do Projeto
Este projeto implementa uma solução completa de Inteligência de Mercado e Previsão de Demanda para o setor farmacêutico. O objetivo principal é demonstrar como aplicar técnicas avançadas de Engenharia de Dados e Machine Learning para transformar dados brutos em insights estratégicos de negócio.

Diferente de abordagens tradicionais que focam puramente em métricas técnicas (como MAE ou RMSE), este sistema prioriza a explicabilidade e o storytelling, traduzindo algoritmos complexos em recomendações acionáveis para tomadores de decisão.

📸 O Dashboard Executivo
O resultado final é um relatório visual que conta a história dos dados, identificando eventos críticos e drivers de venda.

<img width="1543" height="995" alt="Inteligencia" src="https://github.com/user-attachments/assets/767b64db-f759-436b-b4bf-857666b4d6e1" />


🏗️ Arquitetura e Stack Tecnológica
A arquitetura foi desenhada para alta performance e escalabilidade, utilizando ferramentas modernas do ecossistema Python:

• Processamento de Dados: Polars (Substituto moderno do Pandas, otimizado para execução paralela e baixo consumo de memória).

• Machine Learning: LightGBM (Algoritmo de Gradient Boosting de alta eficiência para dados tabulares e séries temporais).

• Visualização de Dados: Matplotlib & Seaborn (Customização avançada para narrativas visuais).

• Engenharia: Pydantic (Validação de dados), Python 3.10+.

💡 Principais Funcionalidades

1. Simulação de Cenários Realistas

O sistema não utiliza dados aleatórios simples. Implementa um Motor de Cenários que simula eventos reais do varejo:

• Sazonalidade: Picos de venda no inverno (gripes/resfriados) e verão.

• Eventos de Marketing: Impacto agressivo da Black Friday.

• Ruptura de Estoque: Simulação de perda de vendas por falta de produto, permitindo testar a resiliência do modelo a anomalias.

2. Pipeline de Feature Engineering

Transformação de dados brutos em features preditivas robustas:

• Criação de Lags (Defasagem de tempo) para captura de dependência temporal.

• Médias Móveis para suavização de tendências.

• Codificação cíclica para variáveis temporais (dia do ano, mês).

3. Foco em Negócio (Storytelling)

O dashboard final traduz o "tecnicês" para o idioma do negócio:

• Diagnóstico de Anomalias: Identificação visual de quedas de faturamento causadas por ruptura de estoque.

• Drivers de Venda: Gráfico de importância das features traduzido (ex: "Vendas Ontem" em vez de "lag_1").

• Zona de Confiança: Visualização clara do intervalo de confiança da previsão para gestão de riscos.

⚙️ Como Executar o Projeto

Pré-requisitos:

• Python 3.9+

• Pip ou Conda

Passo a passo:

Clone o repositório:

git clone https://github.com/Runaway1457/pharma-demand-forecast-system.git

Crie o ambiente virtual e instale as dependências:

bash

pip install -r requirements.txt

Execute o script principal para gerar o relatório:

bash

python main.py

(O script irá gerar a imagem business_storytelling_dashboard.png no diretório local.)

📊 Entendendo os Resultados

O modelo consegue prever a demanda com precisão, mas o grande valor está na detecção de desvios:

• Ao identificar que as vendas caíram em Janeiro/2024, o modelo aponta que isso não foi uma perda natural de demanda, mas uma anomalia (Ruptura), permitindo ação corretiva da operação.

• A análise de resíduos mostra que o erro é distribuído normalmente, validando a robustez estatística do modelo.

📄 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar, modificar e distribuir.

👨‍💻 Autor
Gabriel Borges

LinkedIn: https://www.linkedin.com/in/gabriel-borges25/

GitHub: https://github.com/Runaway1457
