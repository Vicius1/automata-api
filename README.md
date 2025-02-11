# API de Autômatos

Esta API REST foi desenvolvida para manipular e analisar autômatos (AFD, NFA, DPDA, NPDA, TM, NTM, MNTM) utilizando a biblioteca [Automata](https://github.com/caleb531/automata) e o framework [FastAPI](https://fastapi.tiangolo.com/).

## Como Configurar e Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- [pip](https://pip.pypa.io/)
- (Opcional) Um ambiente virtual

### Instalação

1. Clone o repositório:
   
2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux

3. Instale as dependências:

   ```bash
   pip install fastapi uvicorn pydot
   pip install git+https://github.com/caleb531/automata.git

### Execução

1. Para executar:
   
   ```bash
   uvicorn main:app --reload

### Exemplos

Use os exemplos que estão no .pdf
