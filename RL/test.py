import numpy as np

# Configurações do ambiente e hiperparâmetros
num_estados = 5
num_acoes = 2
tau = 0.001          # Fator de escala do bônus de exploração
passos_planejamento = 10

# Inicialização das estruturas do Dyna-Q+
# Q-table: guarda os valores estimados de cada par (estado, ação)
Q_table = np.zeros((num_estados, num_acoes))

# tau_n: guarda o número de passos desde a última visita a cada par (estado, ação)
# No Dyna-Q+, inicializa-se frequentemente com zero ou um valor pequeno
tau_n = np.zeros((num_estados, num_acoes))

# Modelo de transição: guarda (recompensa_real, próximo_estado) para os pares visitados
modelo = {}

def atualizar_contadores_tempo(estado_atual, acao_atual):
    """
    Incrementa o tempo de todos os pares estado-ação em 1,
    e zera o contador do par que acabou de ser visitado no ambiente real.
    """
    global tau_n
    tau_n += 1
    tau_n[estado_atual, acao_atual] = 0

def fase_de_planejamento():
    """
    Simula a fase de planejamento do Dyna-Q+, onde o bônus
    de exploração é adicionado à recompensa do modelo.
    """
    global Q_table
    
    # O Dyna-Q+ permite selecionar até mesmo estados/ações nunca antes visitados
    for _ in range(passos_planejamento):
        # Seleciona aleatoriamente um estado e uma ação qualquer (característica do Dyna-Q+)
        estado_sim = np.random.randint(0, num_estados)
        acao_sim = np.random.randint(0, num_acoes)
        
        # Se o par já foi visitado, pegamos a recompensa e o próximo estado reais.
        # Se nunca foi visitado, o modelo assume transição para o próprio estado e recompensa zero.
        if (estado_sim, acao_sim) in modelo:
            recompensa_modelo, proximo_estado_sim = modelo[(estado_sim, acao_sim)]
        else:
            recompensa_modelo, proximo_estado_sim = 0.0, estado_sim
            
        # CALCULO DO BÔNUS DYNA-Q+
        # R_bonus = tau * sqrt(tau_n)
        tempo_sem_visita = tau_n[estado_sim, acao_sim]
        bonus = tau * np.sqrt(tempo_sem_visita)
        
        # Recompensa modificada usada apenas no planejamento
        recompensa_planejamento = recompensa_modelo + bonus
        
        # Atualização clássica de Q-Learning (fase de planejamento)
        alpha = 0.1
        gama = 0.95
        melhor_proxima_acao = np.max(Q_table[proximo_estado_sim])
        
        Q_table[estado_sim, acao_sim] += alpha * (
            recompensa_planejamento + gama * melhor_proxima_acao - Q_table[estado_sim, acao_sim]
        )

# --- Exemplo de um Passo no Ambiente Real ---

# 1. Agente está no estado 0, escolhe a ação 1 no mundo real
estado_atual = 0
acao_atual = 1

# 2. Transição real (o ambiente devolve uma recompensa e o próximo estado)
recompensa_real = 1.0
proximo_estado = 1

# 3. Atualiza o modelo com a experiência real
modelo[(estado_atual, acao_atual)] = (recompensa_real, proximo_estado)

# 4. Atualiza os contadores de tempo tau_n
atualizar_contadores_tempo(estado_atual, acao_atual)

# 5. Executa o planejamento Dyna-Q+ usando o bônus calculado
fase_de_planejamento()

print("Exemplo de contadores tau_n após um passo (par [0,1] foi zerado):\n", tau_n)
