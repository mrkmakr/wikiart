import pandas as pd
import os
import shutil

bignames = [
'Jan van Eyck',
'Sandro Botticelli',
'Leonardo da Vinci',
'Hieronymus Bosch',
'Albrecht Durer',
'Michelangelo',
'Raphael',
'Giuseppe Arcimboldo',
'Caravaggio',
'Diego Velazquez',
'Rembrandt',
'Johannes Vermeer',
'Francisco Goya',
'William Turner',
'William Turner',
'Jean-Francois Millet',
'John Everett Millais',
'Edouard Manet',
'Edgar Degas',
'Paul Cezanne',
'Claude Monet',
'Pierre-Auguste Renoir',
'Pieter Bruegel the Elder',
'Henri Rousseau',
'Paul Gauguin',
'Georges Seurat',
'Vincent van Gogh',
'Alphonse Mucha',
'Edvard Munch',
'Henri Matisse',
'Piet Mondrian',
'Paul Klee',
'Pablo Picasso',
'Georges Braque',
'Marc Chagall',
'Egon Schiele',
'Joan Miro',
'Mark Rothko',
'Salvador Dali',
'Jackson Pollock',
'Andy Warhol'
]

if __name__ == '__main__':
    data = pd.read_csv('all_data_info.csv', encoding='utf8')
    os.mkdir("./artists")

    for bigname in bignames:
        files = data[data.in_train & (data.artist == bigname)].new_filename.tolist()
        os.mkdir("../artists/{}".format(bigname))
        for file in files:
            shutil.copyfile("../files/{}".format(file), "../artists/{0}/{1}".format(bigname, file))

