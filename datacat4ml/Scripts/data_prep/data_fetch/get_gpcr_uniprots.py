""" Get UNIPROT IDs for GPCRs

1. Go to Uniprot website https://www.uniprot.org/
2. Search 'GPCR' in the seach box.
3. Set the following filters:
    - Status: Reviewed
    - Popular Organisms : Human
    and resulted 898 entries
4. Click 'Share' on the top right of the listing results, and then selec 'Generate URL for API'. The following URLs are obtained:
    - Format: FASTA(canonical), Compressed: Yes, API URL using the streaming endpoint
        
        https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28GPCR%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29&sort=gene+asc

    - Fomat: TSV, Compressed: Yes, API URL using the streaming endpoint
    
        https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=tsv&query=%28GPCR%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29&sort=gene+asc

"""

import os
import requests

# inner module import
import sys
from datacat4ml.Scripts.const import DATA_DIR


def download_url(url, file_name):
    response = requests.get(url)
    with open(file_name, 'wb') as file:
        file.write(response.content)
    os.system('gunzip ' + file_name)

# Download the file from `url` and save it locally:
FASTA_url = 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28GPCR%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29&sort=gene+asc'
TSV_url = 'https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cid%2Cprotein_name%2Cgene_names%2Corganism_name%2Clength&format=tsv&query=%28GPCR%29+AND+%28model_organism%3A9606%29+AND+%28reviewed%3Atrue%29&sort=gene+asc'

# Download the compressed file from `url`, save it locally and decompress it:
download_url(FASTA_url, os.path.join(DATA_DIR, 'GPCR_human.fasta.gz'))
download_url(TSV_url, os.path.join(DATA_DIR, 'GPCR_human.tsv.gz'))