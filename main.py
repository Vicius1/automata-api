import uuid
import tempfile
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import automata.base.exceptions as exceptions
from automata.fa.dfa import DFA
from automata.pda.dpda import DPDA
from automata.tm.dtm import DTM
from automata.fa.nfa import NFA
from automata.pda.npda import NPDA
from automata.tm.ntm import NTM
from automata.tm.mntm import MNTM

import pydot

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API de Autômatos",
    description="API REST para criação, consulta, teste e visualização de autômatos utilizando a biblioteca Automata."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Armazenamento em memória
dfa_storage: Dict[str, DFA] = {}
dpda_storage: Dict[str, DPDA] = {}
tm_storage: Dict[str, DTM] = {}
nfa_storage: Dict[str, NFA] = {}
npda_storage: Dict[str, NPDA] = {}
ntm_storage: Dict[str, NTM] = {}
mntm_storage: Dict[str, MNTM] = {}

# -----------------------------------------------------
# MODELOS (Schemas) para o AFD
# -----------------------------------------------------

class DFAModel(BaseModel):
    states: List[str]
    input_symbols: List[str]
    transitions: Dict[str, Dict[str, str]]
    initial_state: str
    final_states: List[str]

class TestInput(BaseModel):
    input_string: str

# -----------------------------------------------------
# Endpoints para AFD
# -----------------------------------------------------

@app.post("/afd/create", tags=["AFD"], summary="Cria um novo AFD")
def create_afd(dfa_model: DFAModel):
    try:
        dfa = DFA(
            states=set(dfa_model.states),
            input_symbols=set(dfa_model.input_symbols),
            transitions={state: trans for state, trans in dfa_model.transitions.items()},
            initial_state=dfa_model.initial_state,
            final_states=set(dfa_model.final_states)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na criação do AFD: {e}")
    
    dfa_id = str(uuid.uuid4())
    dfa_storage[dfa_id] = dfa
    return {"id": dfa_id, "message": "AFD criado com sucesso"}

@app.get("/afd/{dfa_id}", tags=["AFD"], summary="Consulta informações do AFD")
def get_afd(dfa_id: str):
    if dfa_id not in dfa_storage:
        raise HTTPException(status_code=404, detail="AFD não encontrado")
    
    dfa = dfa_storage[dfa_id]
    return {
        "states": list(dfa.states),
        "input_symbols": list(dfa.input_symbols),
        "transitions": dfa.transitions,
        "initial_state": dfa.initial_state,
        "final_states": list(dfa.final_states)
    }
    

@app.post("/afd/{dfa_id}/test", tags=["AFD"], summary="Testa a aceitação de uma string")
async def test_afd(dfa_id: str, input_data: TestInput):
    """
    Testa se uma determinada string é aceita pelo AFD.
    Retorna uma mensagem informando se a string foi aceita ou não.
    """
    if dfa_id not in dfa_storage:
        raise HTTPException(status_code=404, detail="AFD não encontrado")
    
    dfa = dfa_storage.get(dfa_id)
    try:
        # O método expects uma lista de símbolos (cada caractere) da string de entrada.
        accepted = dfa.accepts_input(list(input_data.input_string))
        if accepted:
            return {"message": "A string foi aceita pelo AFD."}
        else:
            return {"message": "A string não foi aceita pelo AFD."}
    except Exception as e:
        # Se a exceção indicar rejeição, podemos retornar uma mensagem apropriada
        if "RejectionException" in str(e):
            return {"message": "A string não foi aceita pelo AFD."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")


@app.get("/afd/{dfa_id}/visualize", tags=["AFD"], summary="Visualiza o AFD (PNG)")
def visualize_afd(dfa_id: str):
    """
    Gera e retorna uma imagem PNG representando o AFD (utiliza Graphviz via pydot).
    """
    if dfa_id not in dfa_storage:
        raise HTTPException(status_code=404, detail="AFD não encontrado")
    
    dfa = dfa_storage[dfa_id]
    graph = pydot.Dot("dfa", graph_type="digraph")
    
    # Cria os nós (estados)
    for state in dfa.states:
        shape = "doublecircle" if state in dfa.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria as arestas (transições)
    for state, trans in dfa.transitions.items():
        for symbol, next_state in trans.items():
            edge = pydot.Edge(state, next_state, label=symbol)
            graph.add_edge(edge)
    
    # Gera um arquivo temporário com a imagem PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")


# -----------------------------------------------------
# MODELOS (Schemas) para o DPDA
# -----------------------------------------------------

class DPDAConfig(BaseModel):
    states: set
    input_symbols: set
    stack_symbols: set
    transitions: dict
    initial_state: str
    initial_stack_symbol: str
    final_states: set
    acceptance_mode: str

# Definindo a classe DPDA para armazenar a configuração do DPDA
class DPDA:
    def __init__(self, states, input_symbols, stack_symbols, transitions, initial_state, initial_stack_symbol, final_states, acceptance_mode):
        self.states = states
        self.input_symbols = input_symbols
        self.stack_symbols = stack_symbols
        self.transitions = transitions
        self.initial_state = initial_state
        self.initial_stack_symbol = initial_stack_symbol
        self.final_states = final_states
        self.acceptance_mode = acceptance_mode

# -----------------------------------------------------
# Endpoints para DPDA
# -----------------------------------------------------

@app.post("/dpda/create", tags=["DPDA"], summary="Cria um novo DPDA")
async def create_dpda(config: DPDAConfig):
    try:
        # Cria o DPDA com as configurações fornecidas
        dpda = DPDA(
            states=config.states,
            input_symbols=config.input_symbols,
            stack_symbols=config.stack_symbols,
            transitions=config.transitions,
            initial_state=config.initial_state,
            initial_stack_symbol=config.initial_stack_symbol,
            final_states=config.final_states,
            acceptance_mode=config.acceptance_mode
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao criar DPDA: {e}")

    dpda_id = str(uuid.uuid4())
    dpda_storage[dpda_id] = dpda  # Armazena o DPDA criado no dicionário
    return {"id": dpda_id, "message": "DPDA criado com sucesso"}
   


@app.get("/dpda/{dpda_id}", tags=["DPDA"], summary="Consulta informações do DPDA")
async def get_dpda(dpda_id: str):
    if dpda_id not in dpda_storage:
        raise HTTPException(status_code=404, detail="DPDA não encontrado")

    dpda = dpda_storage.get(dpda_id)
    return {
        "states": dpda.states,
        "input_symbols": dpda.input_symbols,
        "stack_symbols": dpda.stack_symbols,
        "transitions": dpda.transitions,
        "initial_state": dpda.initial_state,
        "initial_stack_symbol": dpda.initial_stack_symbol,
        "final_states": dpda.final_states,
        "acceptance_mode": dpda.acceptance_mode
    }


# Função auxiliar que simula a leitura da entrada pelo DPDA.
def read_input(dpda, input_string: str):
    """
    Processa a string de entrada de acordo com as transições do DPDA e retorna
    uma configuração final representada por um dicionário com:
      - state: estado final
      - remaining_input: string que sobrou (deve ser "")
      - stack: a pilha final (lista de símbolos)
      
    Lança uma exceção com a mensagem "RejectionException" se a entrada for rejeitada.
    """
    current_state = dpda.initial_state
    stack = [dpda.initial_stack_symbol]
    
    # Processa cada símbolo da entrada
    for symbol in input_string:
        # Verifica se há transição definida para o estado atual e o símbolo de entrada
        if current_state not in dpda.transitions or symbol not in dpda.transitions[current_state]:
            raise Exception("RejectionException: Transição indefinida para o símbolo '{}' no estado '{}'.".format(symbol, current_state))
        
        # Obtém o topo da pilha (se houver); caso contrário, usa uma string vazia
        top = stack.pop() if stack else ""
        
        # Busca a transição para a combinação (estado, símbolo, topo)
        transition = dpda.transitions[current_state][symbol].get(top)
        if not transition:
            raise Exception("RejectionException: Nenhuma transição válida para (estado: {}, símbolo: '{}', topo: '{}').".format(current_state, symbol, top))
        
        # A transição está no formato: [novo_estado, <símbolos a empilhar...>]
        new_state = transition[0]
        push_symbols = transition[1:]  # pode ser uma lista de 0 ou mais símbolos
        
        # Empilha os símbolos (se houver) na ordem correta:
        # Exemplo: se push_symbols for ["1", "0"], empilhamos "0" primeiro e depois "1"
        for sym in reversed(push_symbols):
            if sym != "":
                stack.append(sym)
        current_state = new_state

    # Após consumir toda a entrada, processa transições lambda (com chave "") enquanto possível.
    # Note: Aqui usamos um laço while; a ideia é aplicar todas as transições vazias disponíveis.
    while "" in dpda.transitions.get(current_state, {}):
        # Obtém o topo da pilha para a transição lambda
        top = stack.pop() if stack else ""
        lambda_transition = dpda.transitions[current_state][""].get(top)
        if not lambda_transition:
            # Se não houver transição válida para a lambda, encerra o laço.
            break
        new_state = lambda_transition[0]
        push_symbols = lambda_transition[1:]
        for sym in reversed(push_symbols):
            if sym != "":
                stack.append(sym)
        current_state = new_state

    # Verifica a aceitação com base no modo de aceitação definido
    if dpda.acceptance_mode == "final_state":
        if current_state in dpda.final_states:
            return {"state": current_state, "remaining_input": "", "stack": stack}
        else:
            raise Exception("RejectionException: Estado final '{}' não está em final_states {}.".format(current_state, dpda.final_states))
    elif dpda.acceptance_mode == "empty_stack":
        if not stack:
            return {"state": current_state, "remaining_input": "", "stack": stack}
        else:
            raise Exception("RejectionException: A pilha não está vazia no final: {}.".format(stack))
    elif dpda.acceptance_mode == "both":
        if current_state in dpda.final_states and not stack:
            return {"state": current_state, "remaining_input": "", "stack": stack}
        else:
            raise Exception("RejectionException: Critério de aceitação não atendido (estado: {}, pilha: {}).".format(current_state, stack))
    else:
        raise Exception("RejectionException: Modo de aceitação inválido.")

# Endpoint para testar a aceitação de uma string pelo DPDA
@app.post("/dpda/{dpda_id}/test", tags=["DPDA"], summary="Testa a aceitação de uma string")
async def test_dpda(dpda_id: str, input_data: TestInput):

    if dpda_id not in dpda_storage:
        raise HTTPException(status_code=404, detail="DPDA não encontrado")

    dpda = dpda_storage.get(dpda_id)
    try:
        # Tenta processar a entrada usando a função read_input
        config = read_input(dpda, input_data.input_string)
        return {"message": "A string foi aceita pelo DPDA.", "configuration": config}
    except Exception as e:
        # Se a exceção indicar rejeição, retornamos a mensagem de rejeição.
        if "RejectionException" in str(e):
            return {"message": "A string não foi aceita pelo DPDA."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")


# Endpoint para visualizar o DPDA em formato PNG
@app.get("/dpda/{dpda_id}/visualize", tags=["DPDA"], summary="Visualiza o DPDA (PNG)")
def visualize_dpda(dpda_id: str):
    """
    Gera e retorna uma imagem PNG representando o DPDA.
    
    Para cada transição, o rótulo é exibido no formato:
      "<input_symbol ou ε>, <topo_da_pilha> | <símbolos_a_empilhar ou ε>"
    """
    if dpda_id not in dpda_storage:
        raise HTTPException(status_code=404, detail="DPDA não encontrado")
    
    dpda = dpda_storage[dpda_id]
    graph = pydot.Dot("dpda", graph_type="digraph")
    
    # Cria os nós (estados)
    for state in dpda.states:
        shape = "doublecircle" if state in dpda.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria as arestas (transições)
    # dpda.transitions é um dicionário com a seguinte estrutura:
    # { estado_origem: { input_symbol: { stack_symbol: [estado_destino, símbolo1, símbolo2, ...] } } }
    for state, transitions_for_state in dpda.transitions.items():
        for input_symbol, transition_dict in transitions_for_state.items():
            for stack_symbol, transition in transition_dict.items():
                # transition é uma lista; o primeiro elemento é o estado destino,
                # os demais elementos são os símbolos a serem empilhados.
                next_state = transition[0]
                push_symbols = transition[1:]
                # Converte a entrada e os símbolos a empilhar para uma string:
                symbol_str = input_symbol if input_symbol != "" else "ε"
                push_str = ", ".join(push_symbols) if any(push_symbols) else "ε"
                label = f"{symbol_str}, {stack_symbol} | {push_str}"
                edge = pydot.Edge(state, next_state, label=label)
                graph.add_edge(edge)
    
    # Gera um arquivo temporário com a imagem PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

# -----------------------------------------------------
# MODELOS (Schemas) para o TM
# -----------------------------------------------------

class TuringMachineConfig(BaseModel):
    states: set
    input_symbols: set
    tape_symbols: set
    transitions: dict
    initial_state: str
    blank_symbol: str
    final_states: set

# Classe TuringMachine
class TuringMachine:
    def __init__(self, states, input_symbols, tape_symbols, transitions, initial_state, blank_symbol, final_states):
        self.states = states
        self.input_symbols = input_symbols
        self.tape_symbols = tape_symbols
        self.transitions = transitions
        self.initial_state = initial_state
        self.blank_symbol = blank_symbol
        self.final_states = final_states
    
    def read_input(self, input_string: str):
        """
        Simula o processamento da fita pela Máquina de Turing.
        
        - A fita é representada como uma lista de símbolos.
        - O cabeçote inicia na posição 0.
        - A simulação é limitada a um número máximo de passos para evitar loops infinitos.
        - Se o autômato atingir um estado final, a função retorna uma configuração final.
        - Caso contrário (ou se não houver transição definida), lança uma exceção do tipo "RejectionException".
        """
        tape = list(input_string)
        head = 0
        current_state = self.initial_state
        steps = 0
        max_steps = 1000  # Limite para evitar loops infinitos
        
        while steps < max_steps:
            steps += 1
            # Expande a fita para a esquerda ou direita conforme necessário
            if head < 0:
                tape.insert(0, self.blank_symbol)
                head = 0
            elif head >= len(tape):
                tape.append(self.blank_symbol)
            
            # Se estivermos em um estado final, aceitamos a entrada
            if current_state in self.final_states:
                return {"state": current_state, "tape": "".join(tape), "head": head}
            
            current_symbol = tape[head]
            
            # Procura a transição para a combinação (estado atual, símbolo lido)
            if current_state in self.transitions and current_symbol in self.transitions[current_state]:
                # Cada transição tem o formato: (next_state, write_symbol, direction)
                next_state, write_symbol, direction = self.transitions[current_state][current_symbol]
                tape[head] = write_symbol
                current_state = next_state
                if direction.upper() == "R":
                    head += 1
                elif direction.upper() == "L":
                    head -= 1
                # Se direction for "S" (stay), o cabeçote não se move.
            else:
                raise Exception("RejectionException: Transição indefinida para (estado: {}, símbolo: {}).".format(current_state, current_symbol))
        
        # Se excedermos o número máximo de passos, consideramos que a máquina não halta (rejeita a entrada)
        raise Exception("RejectionException: Excedido o número máximo de passos.")

# -----------------------------------------------------
# Endpoints para TM
# -----------------------------------------------------

# Endpoint para criar uma Máquina de Turing
@app.post("/tm/create", tags=["Turing Machine"], summary="Cria uma nova Máquina de Turing")
async def create_tm(config: TuringMachineConfig):
    try:
        tm = TuringMachine(
            states=config.states,
            input_symbols=config.input_symbols,
            tape_symbols=config.tape_symbols,
            transitions=config.transitions,
            initial_state=config.initial_state,
            blank_symbol=config.blank_symbol,
            final_states=config.final_states
        )   
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao criar a Máquina de Turing: {e}")
    tm_id = str(uuid.uuid4())
    tm_storage[tm_id] = tm
    return {"id": tm_id, "message": "TM criado com sucesso"}

# Endpoint para consultar informações de uma Máquina de Turing
@app.get("/tm/{tm_id}", tags=["Turing Machine"], summary="Consulta informações da Máquina de Turing")
async def get_tm(tm_id: str):
    if tm_id not in tm_storage:
        raise HTTPException(status_code=404, detail="Máquina de Turing não encontrada")

    tm = tm_storage.get(tm_id)
    return {
        "states": tm.states,
        "input_symbols": tm.input_symbols,
        "tape_symbols": tm.tape_symbols,
        "transitions": tm.transitions,
        "initial_state": tm.initial_state,
        "blank_symbol": tm.blank_symbol,
        "final_states": tm.final_states
    }

# Endpoint para testar uma entrada na Máquina de Turing
@app.post("/tm/{tm_id}/test", tags=["Turing Machine"], summary="Testa a entrada na Máquina de Turing")
async def test_tm(tm_id: str, input_data: TestInput):
    if tm_id not in tm_storage:
        raise HTTPException(status_code=404, detail="Máquina de Turing não encontrada")

    tm = tm_storage.get(tm_id)
    try:
        config = tm.read_input(input_data.input_string)
        return {"message": "A entrada foi aceita pela Máquina de Turing.", "configuration": config}
    except Exception as e:
        if "RejectionException" in str(e):
            return {"message": "A entrada não foi aceita pela Máquina de Turing."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")

# Endpoint para visualizar a Máquina de Turing (PNG)
@app.get("/tm/{tm_id}/visualize", tags=["Turing Machine"], summary="Visualiza a Máquina de Turing (PNG)")
def visualize_tm(tm_id: str):
    if tm_id not in tm_storage:
        raise HTTPException(status_code=404, detail="Máquina de Turing não encontrada")
    tm = tm_storage[tm_id]
    graph = pydot.Dot("tm", graph_type="digraph")
    
    # Cria nós para cada estado; estados finais são representados com "doublecircle"
    for state in tm.states:
        shape = "doublecircle" if state in tm.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria arestas para as transições
    # Assume-se que as transições estão no formato:
    # { estado: { símbolo_lido: (estado_destino, símbolo_escrito, direção) } }
    for state, trans in tm.transitions.items():
        for symbol, action in trans.items():
            next_state, write_symbol, direction = action
            # Formata o rótulo: "símbolo_lido | símbolo_escrito | direção"
            label = f"{symbol} | {write_symbol} | {direction}"
            edge = pydot.Edge(state, next_state, label=label)
            graph.add_edge(edge)
    
    # Gera um arquivo temporário com a imagem PNG
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

# -----------------------------------------------------
# MODELOS (Schemas) para o NFA
# -----------------------------------------------------

class NFAModel(BaseModel):
    states: List[str]
    input_symbols: List[str]
    transitions: Dict[str, Dict[str, List[str]]]
    initial_state: str
    final_states: List[str]

# -----------------------------------------------------
# Endpoints para TM
# -----------------------------------------------------

@app.post("/nfa/create", tags=["NFA"], summary="Cria um novo NFA")
def create_nfa(nfa_model: NFAModel):
    """
    Cria um NFA (Autômato Finito Não Determinístico) com base na configuração enviada.
    Converte as listas de estados de destino em conjuntos.
    """
    try:
        # Converte a configuração para os tipos esperados pelo NFA
        transitions_converted = {}
        for state, trans in nfa_model.transitions.items():
            transitions_converted[state] = {}
            for symbol, dest_list in trans.items():
                transitions_converted[state][symbol] = set(dest_list)
                
        nfa = NFA(
            states=set(nfa_model.states),
            input_symbols=set(nfa_model.input_symbols),
            transitions=transitions_converted,
            initial_state=nfa_model.initial_state,
            final_states=set(nfa_model.final_states)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na criação do NFA: {e}")
    
    nfa_id = str(uuid.uuid4())
    nfa_storage[nfa_id] = nfa
    return {"id": nfa_id, "message": "NFA criado com sucesso"}

@app.get("/nfa/{nfa_id}", tags=["NFA"], summary="Consulta informações do NFA")
def get_nfa(nfa_id: str):
    if nfa_id not in nfa_storage:
        raise HTTPException(status_code=404, detail="NFA não encontrado")
    
    nfa = nfa_storage[nfa_id]
    # Converte os conjuntos de transições para listas para que o JSON seja serializável
    transitions_serializable = {}
    for state, trans in nfa.transitions.items():
        transitions_serializable[state] = {}
        for symbol, dest_set in trans.items():
            transitions_serializable[state][symbol] = list(dest_set)
    
    return {
        "states": list(nfa.states),
        "input_symbols": list(nfa.input_symbols),
        "transitions": transitions_serializable,
        "initial_state": nfa.initial_state,
        "final_states": list(nfa.final_states)
    }

@app.post("/nfa/{nfa_id}/test", tags=["NFA"], summary="Testa a aceitação de uma string")
def test_nfa(nfa_id: str, input_data: TestInput):
    if nfa_id not in nfa_storage:
        raise HTTPException(status_code=404, detail="NFA não encontrado")
    
    nfa = nfa_storage[nfa_id]
    try:
        # Converte a string de entrada em lista de símbolos e testa a aceitação
        accepted = nfa.accepts_input(list(input_data.input_string))
        if accepted:
            return {"message": "A string foi aceita pelo NFA."}
        else:
            return {"message": "A string não foi aceita pelo NFA."}
    except Exception as e:
        if "RejectionException" in str(e):
            return {"message": "A string não foi aceita pelo NFA."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")

@app.get("/nfa/{nfa_id}/visualize", tags=["NFA"], summary="Visualiza o NFA (PNG)")
def visualize_nfa(nfa_id: str):
    if nfa_id not in nfa_storage:
        raise HTTPException(status_code=404, detail="NFA não encontrado")
    
    nfa = nfa_storage[nfa_id]
    graph = pydot.Dot("nfa", graph_type="digraph")
    
    # Cria os nós (estados)
    for state in nfa.states:
        shape = "doublecircle" if state in nfa.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria as arestas (transições)
    # transitions: { state: { symbol: set(dest_states) } }
    for state, trans in nfa.transitions.items():
        for symbol, dest_set in trans.items():
            # Se a transição for lambda (vazia), usamos o símbolo ε
            symbol_str = symbol if symbol != "" else "ε"
            for dest in dest_set:
                edge = pydot.Edge(state, dest, label=symbol_str)
                graph.add_edge(edge)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

# -----------------------------------------------------
# MODELOS (Schemas) para o NPDA
# -----------------------------------------------------

class NPDAConfig(BaseModel):
    states: set
    input_symbols: set
    stack_symbols: set
    transitions: dict
    initial_state: str
    initial_stack_symbol: str
    final_states: set
    acceptance_mode: str

# -----------------------------------------------------
# Endpoints para NPDA
# -----------------------------------------------------

@app.post("/npda/create", tags=["NPDA"], summary="Cria um novo NPDA")
def create_npda(config: NPDAConfig):
    """
    Cria um NPDA (Autômato com Pilha Não Determinístico) com base na configuração enviada.
    As listas de destinos (dentro de transitions) serão convertidas em conjuntos de tuplas,
    onde cada tupla tem o formato (next_state, push_symbols).
    
    Se a transição for de pop (ou seja, apenas desempilhar), o push_symbols será uma string vazia.
    """
    try:
        transitions_converted = {}
        for state, trans in config.transitions.items():
            transitions_converted[state] = {}
            for symbol, mapping in trans.items():
                transitions_converted[state][symbol] = {}
                for stack_sym, dest_list in mapping.items():
                    converted_destinations = set()
                    for item in dest_list:
                        if isinstance(item, (list, tuple)):
                            if len(item) >= 2:
                                next_state = item[0]
                                # Se a transição tiver exatamente 2 elementos e o segundo for "",
                                # usa-se "" em vez de ("",)
                                if len(item[1:]) == 1 and item[1] == "":
                                    push_symbols = ""
                                else:
                                    push_symbols = tuple(item[1:])
                            else:
                                raise Exception(f"Transição inválida: esperado pelo menos 2 elementos, obtido: {item}")
                            converted_destinations.add((next_state, push_symbols))
                        else:
                            raise Exception(f"Item de transição deve ser lista ou tuplo, obtido: {item}")
                    transitions_converted[state][symbol][stack_sym] = converted_destinations

        npda = NPDA(
            states=set(config.states),
            input_symbols=set(config.input_symbols),
            stack_symbols=set(config.stack_symbols),
            transitions=transitions_converted,
            initial_state=config.initial_state,
            initial_stack_symbol=config.initial_stack_symbol,
            final_states=set(config.final_states),
            acceptance_mode=config.acceptance_mode
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na criação do NPDA: {e}")
    
    npda_id = str(uuid.uuid4())
    npda_storage[npda_id] = npda
    return {"id": npda_id, "message": "NPDA criado com sucesso"}


@app.get("/npda/{npda_id}", tags=["NPDA"], summary="Consulta informações do NPDA")
def get_npda(npda_id: str):
    if npda_id not in npda_storage:
        raise HTTPException(status_code=404, detail="NPDA não encontrado")
    
    npda = npda_storage[npda_id]
    # Para serializar as transições, convertemos conjuntos para listas.
    transitions_serializable = {}
    for state, trans in npda.transitions.items():
        transitions_serializable[state] = {}
        for symbol, mapping in trans.items():
            transitions_serializable[state][symbol] = {}
            for stack_sym, dest_set in mapping.items():
                transitions_serializable[state][symbol][stack_sym] = [list(item) if isinstance(item, (tuple, list)) else item for item in dest_set]
    
    return {
        "states": list(npda.states),
        "input_symbols": list(npda.input_symbols),
        "stack_symbols": list(npda.stack_symbols),
        "transitions": transitions_serializable,
        "initial_state": npda.initial_state,
        "initial_stack_symbol": npda.initial_stack_symbol,
        "final_states": list(npda.final_states),
        "acceptance_mode": npda.acceptance_mode
    }

@app.post("/npda/{npda_id}/test", tags=["NPDA"], summary="Testa a aceitação de uma string pelo NPDA")
def test_npda(npda_id: str, input_data: TestInput):
    if npda_id not in npda_storage:
        raise HTTPException(status_code=404, detail="NPDA não encontrado")
    
    npda = npda_storage[npda_id]
    try:
        # Utiliza o método read_input para processar toda a entrada
        config = npda.read_input(input_data.input_string)
        # Se não ocorrer exceção, a entrada foi aceita.
        return {"message": "A string foi aceita pelo NPDA.", "configuration": str(config)}
    except Exception as e:
        # Se a mensagem da exceção contiver "RejectionException", trata como rejeição.
        if "RejectionException" in str(e):
            return {"message": "A string não foi aceita pelo NPDA."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")

@app.get("/npda/{npda_id}/visualize", tags=["NPDA"], summary="Visualiza o NPDA (PNG)")
def visualize_npda(npda_id: str):
    if npda_id not in npda_storage:
        raise HTTPException(status_code=404, detail="NPDA não encontrado")
    
    npda = npda_storage[npda_id]
    graph = pydot.Dot("npda", graph_type="digraph")
    
    # Cria os nós (estados)
    for state in npda.states:
        shape = "doublecircle" if state in npda.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria as arestas (transições)
    # Estrutura das transições: { state: { symbol: { stack_symbol: set( (next_state, push_symbols) ) } } }
    for state, trans in npda.transitions.items():
        for symbol, mapping in trans.items():
            # Se o símbolo for vazio, usamos ε
            symbol_str = symbol if symbol != "" else "ε"
            for stack_sym, dest_set in mapping.items():
                for transition in dest_set:
                    # Cada transição: (next_state, push_symbols)
                    next_state = transition[0]
                    push_symbols = transition[1]
                    if isinstance(push_symbols, (list, tuple)):
                        push_str = ", ".join(push_symbols) if push_symbols else "ε"
                    else:
                        push_str = push_symbols if push_symbols != "" else "ε"
                    label = f"{symbol_str}, {stack_sym} | {push_str}"
                    edge = pydot.Edge(state, next_state, label=label)
                    graph.add_edge(edge)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

# -----------------------------------------------------
# MODELOS (Schemas) para o NPDA
# -----------------------------------------------------

class NTMModel(BaseModel):
    states: List[str]
    input_symbols: List[str]
    tape_symbols: List[str]
    transitions: Dict[str, Dict[str, List[List[str]]]]
    initial_state: str
    blank_symbol: str
    final_states: List[str]

# -----------------------------------------------------
# Endpoints para NPDA
# -----------------------------------------------------

@app.post("/ntm/create", tags=["NTM"], summary="Cria uma nova NTM")
def create_ntm(ntm_model: NTMModel):
    """
    Cria uma NTM (Máquina de Turing Não Determinística) com base na configuração enviada.
    As transições são convertidas do formato JSON (listas) para conjuntos de tuplas.
    """
    try:
        transitions_converted = {}
        for state, trans in ntm_model.transitions.items():
            transitions_converted[state] = {}
            for symbol, dest_list in trans.items():
                transitions_converted[state][symbol] = set()
                for item in dest_list:
                    # Cada item deve ser uma lista com 3 elementos: [next_state, write_symbol, direction]
                    if isinstance(item, list) and len(item) == 3:
                        transitions_converted[state][symbol].add(tuple(item))
                    else:
                        raise Exception(f"Transição inválida para {state} com símbolo '{symbol}': {item}")
                        
        ntm = NTM(
            states=set(ntm_model.states),
            input_symbols=set(ntm_model.input_symbols),
            tape_symbols=set(ntm_model.tape_symbols),
            transitions=transitions_converted,
            initial_state=ntm_model.initial_state,
            blank_symbol=ntm_model.blank_symbol,
            final_states=set(ntm_model.final_states)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na criação da NTM: {e}")
    
    ntm_id = str(uuid.uuid4())
    ntm_storage[ntm_id] = ntm
    return {"id": ntm_id, "message": "NTM criado com sucesso"}

@app.get("/ntm/{ntm_id}", tags=["NTM"], summary="Consulta informações da NTM")
def get_ntm(ntm_id: str):
    if ntm_id not in ntm_storage:
        raise HTTPException(status_code=404, detail="NTM não encontrado")
    
    ntm = ntm_storage[ntm_id]
    # Converte os conjuntos de transições para listas para serialização JSON.
    transitions_serializable = {}
    for state, trans in ntm.transitions.items():
        transitions_serializable[state] = {}
        for symbol, trans_set in trans.items():
            transitions_serializable[state][symbol] = [list(t) for t in trans_set]
    
    return {
        "states": list(ntm.states),
        "input_symbols": list(ntm.input_symbols),
        "tape_symbols": list(ntm.tape_symbols),
        "transitions": transitions_serializable,
        "initial_state": ntm.initial_state,
        "blank_symbol": ntm.blank_symbol,
        "final_states": list(ntm.final_states)
    }

@app.post("/ntm/{ntm_id}/test", tags=["NTM"], summary="Testa a entrada na NTM")
def test_ntm(ntm_id: str, input_data: TestInput):
    if ntm_id not in ntm_storage:
        raise HTTPException(status_code=404, detail="NTM não encontrado")
    
    ntm = ntm_storage[ntm_id]
    try:
        final_configurations = None
        # Itera sobre as configurações geradas pelo método read_input_stepwise.
        for configurations in ntm.read_input_stepwise(input_data.input_string):
            final_configurations = configurations
        # Se nenhuma configuração foi produzida, a entrada é rejeitada.
        if final_configurations is None:
            return {"message": "A entrada não foi aceita pela NTM."}
        
        # Verifica se alguma configuração possui um estado final.
        accepted = any(config.state in ntm.final_states for config in final_configurations)
        if accepted:
            return {"message": "A entrada foi aceita pela NTM.", "configuration": str(final_configurations)}
        else:
            return {"message": "A entrada não foi aceita pela NTM."}
    except Exception as e:
        if "RejectionException" in str(e):
            return {"message": "A entrada não foi aceita pela NTM."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")

@app.get("/ntm/{ntm_id}/visualize", tags=["NTM"], summary="Visualiza a NTM (PNG)")
def visualize_ntm(ntm_id: str):
    if ntm_id not in ntm_storage:
        raise HTTPException(status_code=404, detail="NTM não encontrado")
    
    ntm = ntm_storage[ntm_id]
    graph = pydot.Dot("ntm", graph_type="digraph")
    
    # Cria os nós (estados)
    for state in ntm.states:
        shape = "doublecircle" if state in ntm.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria as arestas (transições)
    # A estrutura das transições é: 
    # { state: { symbol: set( (next_state, write_symbol, direction) ) } }
    for state, trans in ntm.transitions.items():
        for symbol, trans_set in trans.items():
            symbol_str = symbol if symbol != "" else "ε"
            for transition in trans_set:
                next_state = transition[0]
                write_symbol = transition[1]
                direction = transition[2]
                label = f"{symbol_str} | {write_symbol} | {direction}"
                edge = pydot.Edge(state, next_state, label=label)
                graph.add_edge(edge)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

# -----------------------------------------------------
# MODELOS (Schemas) para o MNTM
# -----------------------------------------------------

class MNTMModel(BaseModel):
    states: List[str]
    input_symbols: List[str]
    tape_symbols: List[str]
    n_tapes: int
    transitions: Dict[str, Dict[str, List[List]]]
    initial_state: str
    blank_symbol: str
    final_states: List[str]

# -----------------------------------------------------
# Endpoints para MNTM
# -----------------------------------------------------

from automata.tm.mntm import MNTM  # Certifique-se de que essa importação esteja correta

@app.post("/mntm/create", tags=["MNTM"], summary="Cria uma nova MNTM")
def create_mntm(mntm_model: MNTMModel):
    """
    Cria uma MNTM (Máquina de Turing Não Determinística Multitape) com base na configuração enviada.
    Converte os dados do JSON (que vêm como listas e strings) para os tipos esperados pela classe MNTM.
    
    Para as transições, espera-se que as chaves sejam strings no formato "s1,s2,...,sN"
    (por exemplo, "1,#" para uma MNTM de 2 fitas) e que cada valor seja uma lista de transições,
    onde cada transição é do formato:
      [ next_state, [ [write_symbol1, direction1], [write_symbol2, direction2], ... ] ]
    """
    try:
        # Converte as transições
        transitions_converted = {}
        for state, trans in mntm_model.transitions.items():
            transitions_converted[state] = {}
            for key_str, trans_list in trans.items():
                # Converte a chave "s1,s2,...,sN" para tuplo
                key_tuple = tuple(key_str.split(","))
                transitions_converted[state][key_tuple] = set()
                for item in trans_list:
                    # item deve ser uma lista com 2 elementos: [next_state, instructions]
                    if isinstance(item, list) and len(item) == 2:
                        next_state = item[0]
                        instructions_raw = item[1]
                        if not isinstance(instructions_raw, list) or len(instructions_raw) != mntm_model.n_tapes:
                            raise Exception(f"Instruções inválidas para a transição em {state} com chave {key_str}: {item}")
                        instructions_converted = []
                        for instr in instructions_raw:
                            if isinstance(instr, list) and len(instr) == 2:
                                instructions_converted.append(tuple(instr))
                            else:
                                raise Exception(f"Instrução inválida: {instr}")
                        instructions_converted = tuple(instructions_converted)
                        transitions_converted[state][key_tuple].add((next_state, instructions_converted))
                    else:
                        raise Exception(f"Transição inválida: {item}")
        mntm = MNTM(
            states=set(mntm_model.states),
            input_symbols=set(mntm_model.input_symbols),
            tape_symbols=set(mntm_model.tape_symbols),
            n_tapes=mntm_model.n_tapes,
            transitions=transitions_converted,
            initial_state=mntm_model.initial_state,
            blank_symbol=mntm_model.blank_symbol,
            final_states=set(mntm_model.final_states)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na criação da MNTM: {e}")
    
    mntm_id = str(uuid.uuid4())
    mntm_storage[mntm_id] = mntm
    return {"id": mntm_id, "message": "MNTM criada com sucesso"}

@app.get("/mntm/{mntm_id}", tags=["MNTM"], summary="Consulta informações da MNTM")
def get_mntm(mntm_id: str):
    if mntm_id not in mntm_storage:
        raise HTTPException(status_code=404, detail="MNTM não encontrada")
    
    mntm = mntm_storage[mntm_id]
    # Converte os conjuntos de transições para listas para serialização
    transitions_serializable = {}
    for state, trans in mntm.transitions.items():
        transitions_serializable[state] = {}
        for key_tuple, trans_set in trans.items():
            key_str = ", ".join(key_tuple)
            transitions_serializable[state][key_str] = [ [t[0], list(t[1])] for t in trans_set ]
    
    return {
        "states": list(mntm.states),
        "input_symbols": list(mntm.input_symbols),
        "tape_symbols": list(mntm.tape_symbols),
        "n_tapes": mntm.n_tapes,
        "transitions": transitions_serializable,
        "initial_state": mntm.initial_state,
        "blank_symbol": mntm.blank_symbol,
        "final_states": list(mntm.final_states)
    }

@app.post("/mntm/{mntm_id}/test", tags=["MNTM"], summary="Testa a entrada na MNTM")
def test_mntm(mntm_id: str, input_data: TestInput):
    if mntm_id not in mntm_storage:
        raise HTTPException(status_code=404, detail="MNTM não encontrada")
    
    mntm = mntm_storage[mntm_id]
    try:
        final_configs = None
        for config in mntm.read_input_as_ntm(input_data.input_string):
            final_configs = config
        if final_configs is None:
            return {"message": "A entrada não foi aceita pela MNTM."}
        # Verifica se alguma configuração final tem estado final
        accepted = any(cfg.state in mntm.final_states for cfg in final_configs)
        if accepted:
            return {"message": "A entrada foi aceita pela MNTM.", "configuration": str(final_configs)}
        else:
            return {"message": "A entrada não foi aceita pela MNTM."}
    except Exception as e:
        if "RejectionException" in str(e):
            return {"message": "A entrada não foi aceita pela MNTM."}
        else:
            raise HTTPException(status_code=400, detail=f"Erro ao testar a entrada: {e}")

@app.get("/mntm/{mntm_id}/visualize", tags=["MNTM"], summary="Visualiza a MNTM (PNG)")
def visualize_mntm(mntm_id: str):
    if mntm_id not in mntm_storage:
        raise HTTPException(status_code=404, detail="MNTM não encontrada")
    
    mntm = mntm_storage[mntm_id]
    graph = pydot.Dot("mntm", graph_type="digraph")
    
    # Cria nós para cada estado
    for state in mntm.states:
        shape = "doublecircle" if state in mntm.final_states else "circle"
        node = pydot.Node(state, shape=shape)
        graph.add_node(node)
    
    # Cria arestas para as transições
    for state, trans in mntm.transitions.items():
        for key_tuple, trans_set in trans.items():
            key_str = ", ".join(key_tuple) if key_tuple else "ε"
            for transition in trans_set:
                next_state = transition[0]
                instructions = transition[1]  # É um tuplo de instruções, um por fita
                instr_strs = []
                for instr in instructions:
                    # instr é uma tupla (write_symbol, direction)
                    instr_strs.append(f"{instr[0]},{instr[1]}")
                instr_str = "; ".join(instr_strs)
                label = f"{key_str} | {instr_str}"
                edge = pydot.Edge(state, next_state, label=label)
                graph.add_edge(edge)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        graph.write_png(tmp_file.name)
        tmp_file.seek(0)
        image_data = tmp_file.read()
    
    return Response(content=image_data, media_type="image/png")

