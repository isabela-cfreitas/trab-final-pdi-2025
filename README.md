# **Trabalho Final: Aplicações - INF 394 - Processamento Digital de Imagens**

---

## **Informações gerais**

- **Data de entrega**: 04/12/2025 (quinta-feira), **até às 13:00h**
- **Apresentação**: 04/12/2025, a partir das 16:00h
- **Pontuação**: 40 pontos
- **Equipes**: duplas

---

## **Sobre os arquivos e como gerenciá-los**

- Devido à quantidade e tamanho dos arquivos, a disponibilização do material pelo meio que usamos nas aulas práticas (Google Drive) não se mostra a maneira mais adequada, pois demandaria renomeação em massa ou dos arquivos, ou de suas referências no código do enunciado.
- Assim, a exemplo do trabalho anterior, optou-se pelo uso do Github:

    1. Clone o repositório para uma pasta que desejar ([Mini Guia de Uso do Git e GitHub](GuiaGithub.md))
    2. Desenvolva o trabalho (mais sobre isso adiante)
    3. Entregue o Notebook via Moodle com as saídas limpas (instruções detalhadas no final do Notebook)

- Com o repositório clonado, vocês têm as seguintes opções:

    1. Trabalhar localmente, usando uma instalação do `Jupyter` (com `jupyter lab`, de preferência)
    2. Subir o conteúdo todo, respeitando a hierarquia de pastas para o seu Google Drive e utilizar no Google Colab

- A saber:
    - O trabalho todo só depende de 4 bibliotecas Python: `Numpy`, `PIL` (`pillow`), `Matplotlib` e `os` (que é nativa).
    - É bastante fácil gerar um **ambiente virtual** (com `venv`, `conda`, `micromamba` etc.) que contenha as bibliotecas mencionadas acima, acrescidas do Jupyter (com `lab`).
    - Vantagens de se rodar localmente:
        - Sem limite de tempo de execução
        - Sincronização com Github, caso deseje manter em um repositório próprio e integrar entre a dupla de uma maneira mais gerenciável
        - A depender de seu *setup*, ele pode ser mais poderoso que a estrutura disponibilizada no Colab.
    - Desvantagens:
        - Tem que se garantir que as dependências estejam instaladas
        - Para evitar conflitos com outros cenários de desenvolvimento, pode ser que o uso de ambientes virtuais seja necessário
    - Vantagens de rodar no Colab:
        - Sem necessidade de instalação
        - Naturalmente na nuvem, sem demanda de *commits* e operações afins
        - Compartilhamento entre a dupla em poucos cliques
    - Desvantagens:
        - Duas pessoas trabalhando ao mesmo tempo em um arquivo pode gerar conflitos chatos de resolver
        - Sujeito às limitações de hardware impostas pela Google
        - Pode haver necessidade de constante renomeação de arquivos, caso cada membro da dupla deseje manter uma cópia própria do trabalho

- A ponderação acima é apenas para que vocês poderem a respeito dessa decisão. Para o professor, apenas importa o arquivo .ipynb enviado como resposta ao trabalho.

---

## **Sobre o trabalho**

1. Não se assuste com o **tamanho** do Notebook. Os enunciados primam por tentar esclarecer o que está sendo pedido e ilustrar o máximo possível, quando são dados exemplos. Tudo com objetivo de **prevenir dúvidas** de sua parte. Em vários casos, o enunciado será muito maior que o código da solução pedida... 😜

2. O trabalho é factível dentro do prazo dado (aprox. 3 semanas), sem pesar muito a carga dos(as) estudantes. No entanto, se deixar para ser feito na última semana ou últimos dias do prazo, aí sim o trabalho pode ficar grande demais. 

3. O roteiro do trabalho é o próprio notebook. Siga as instruções das células de texto e faça seus códigos nas células Python.

4. Vocês são livres para usar tanto o Google Colab quanto uma máquina Jupyter, localmente, para desenvolver a atividade.

5. O trabalho é composto por várias tarefas individuais, que não são necessariamente conectadas umas às outras. Cada tarefa tem seu próprio enunciado.

6. Recomenda-se fazer os exercícios na ordem que se encontram, pois embora as tarefas não estejam necessariamente conectadas umas às outras, em alguns casos estarão. Quando isso acontecer, estará explícito no enunciado.

7. Está **vetado** o uso de bibliotecas e implementações prontas dos recursos pedidos nas tarefas deste trabalho. Você deve implementar o recurso pedido. A menos que seja dito explicitamente o contrário.

    1. Exemplo 1: se pedido, como alvo da tarefa, que você implemente/faça a conversão de cinza de uma imagem, você não pode usar a conversão que vem pronta na biblioteca PIL

    1. Exemplo 2: se um passo secundário de um algoritmo maior demanda a conversão em cinza de uma imagem, mas não é o foco do exercício, você pode usar a conversão pronta da biblioteca PIL

8. Está **liberado** o uso de outras bibliotecas que desejar incluir, em especial, as não diretamente relacionadas ao conteúdo de PDI.

9. Caso venha a utilizar alguma solução diferente das vistas em aula e/ou fornecidas no material do professor, coloque a fonte (livro, site etc.) de onde foi tirada sua solução.

---

## **Sobre possíveis erros**

- É pouco provável, mas pode ser que algum erro tenha passado despercebido na revisão feita pelo professor no enunciado de cada atividade, como erros de digitação, extensões de arquivos trocadas (ex.: .jpg por .png e vice-versa) e detalhes menores.
- Caso encontre algum erro como os mencionados acima e similares, pode-se utilizar o bom senso e assumir a hipótese mais provável e seguir com o trabalho, sem necessidade de confirmação. Por exemplo, em um exercício hipotético pede-se para que se trabalhe com o arquivo `imagens/rio.png`, mas este arquivo não existe, o que existe é `imagens/rio.jpg`. Altere o código para utilizar o que existe e siga com o desenvolvimento.
- Em casos de dúvidas que envolvam interpretação dos enunciados, tomadas de decisões, inconsistências do que está sendo pedido etc., aí, sim, procure pelo professor para sanar suas dúvidas.

---

## **Dicas úteis**

- Cuidado com casting de tipos, especialmente para `np.uint8`
- Cuidado com operações de divisão (divisão tradicional, inteira ou resto)
- Cuidado com erros de arredondamento. Só arredonde valores no momento de gerar uma imagem visualizável
- Cuidado com formatos das estruturas de dados: $(x,y)$ (geometria) *versus* $(i,j)$ (matriz)
- Lembre-se de clonar o repositório todo e não baixar só o notebook
