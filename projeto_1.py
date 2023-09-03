from sklearn.svm  import LinearSVC
from sklearn.metrics  import accuracy_score

# ==== Features (1 sim, 0 não)
#- Longo?
#- Perna curta?
#- Faz auau?

porco1 = [0, 1 ,0]
porco2 = [0, 1 ,1]
porco3 = [1, 1 ,0]

cachorro1 = [0, 1 ,1]
cachorro2 = [1, 0 ,1]
cachorro3 = [1, 1, 1]


# 1 => porco, 0 => cachorro.
treino_x = [porco1, porco2,  porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1,1,1,0,0,0] # Labels ou Etiquetas

#Cria modelo que irá aprendder.
model = LinearSVC(dual=True)
# Aqui ira absorver os treino_x no aprendizado supervisionado..
model.fit(treino_x, treino_y)


# Tenta prever qual é o animal usando seu algoritomos de previsão
animal_misterioso = [1,1,1]
print(rf'Previsão supervisionada: {model.predict([animal_misterioso])}')


# Tenta prever qual é o animal usando seu algoritomos de previsão
animal_misterioso_1 = [1,1,1]
animal_misterioso_2 = [1,1,0]
animal_misterioso_3 = [0,1,1]
testes = [animal_misterioso_1,animal_misterioso_2,animal_misterioso_3]
o_que_e_de_fato = [0,1,1]
previsoes = model.predict(testes)
print(rf'Previsão de mais de um: {previsoes}')
print(rf'O que é de fato do mais de um: {o_que_e_de_fato}')
# Comparacao do que era com o que previu
print(rf'Comparação do real vs previsão "ACERTOS": {previsoes == o_que_e_de_fato}')
# Soma número de verdadeiros 
# Possibilitado pela biblioteca numpy que facilita
print(rf'Soma dos verdadeiros: {(previsoes == o_que_e_de_fato).sum()}')

# ESTUDO Taxa de acerto:
corretos = (previsoes == o_que_e_de_fato).sum()
total = len(o_que_e_de_fato)
taxa_de_acerto = corretos/total
print(rf"Taxa de acerto cálculo do python: {taxa_de_acerto* 100:.2f} %")

# ESTUDO Taxa de acerto do sklearn:
taxa_de_acerto = accuracy_score(o_que_e_de_fato,previsoes)
print(rf"Taxa de acerto cálculo do sklearn: {taxa_de_acerto* 100:.2f} %")





