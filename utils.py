#### Les fonctions utiles ici 
import pandas as pd 
from pathlib import Path


def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path+file_name)
    
def read_csv(path):
    return pd.read_csv(path)
    
if __name__ == '__main__':
    write_csv([[1,2,3],[4,5,6]], 'ready_cluster/piche/test.csv')