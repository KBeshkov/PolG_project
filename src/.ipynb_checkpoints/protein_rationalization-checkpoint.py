#Protein optimization playground
import rmsd
import itertools
import numpy as np
import subprocess
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)


class ProtRatio:
    def __init__(self,sequence,save_path):
        self.sequence = self.to_elements(sequence)
        self.protein_index = 0
        self.save_path = save_path
        self.structure_original = self.structure_prediciton(self.sequence)
        
    def to_elements(self, sequence):
        return [char for char in sequence]
    
    def to_list(self, sequence):
        return ["".join(sequence)]
    
    def structure_prediciton(self, protein):
        self.save_fasta(*self.to_list(protein), self.save_path+str(self.protein_index)+'.fasta')
        subprocess.run(['colabfold_batch', self.save_path+str(self.protein_index)+'.fasta', self.save_path+'out'])
        struct_file = parser.get_structure("protein_structure", self.save_path+'out/'+"seq1_unrelaxed_rank_005_alphafold2_ptm_model_5_seed_000.pdb")
        print(struct_file)
        return struct_file
    
    def distance_function(self,x):
        return rmsd.rmsd(self.structure_original,x)
    
    def step(self, protein, deletions=1):
        simplices = []
        proteins = []
        scores = []
        for i, comb in enumerate(itertools.combinations(np.arange(0,len(protein)), deletions)):
            simplices.append(comb)
            self.protein_index+=1
            proteins.append(np.delete(protein,comb))
            scores.append(self.distance_function(self.structure_original,self.structure_prediciton(proteins[-1])))
        return proteins[np.argmin(scores)]
    
    def save_fasta(self, sequence, name):
        with open(name, "w") as fasta_file:
            fasta_file.write(f">seq1\n{sequence}")
    
    def rationalize(self, n_steps = 10):
        new_protein = self.sequence.copy()
        protein_trajectory = [new_protein]
        for n in range(n_steps):
            new_protein = self.step(new_protein)
            protein_trajectory.append(self.to_list(new_protein))
        return protein_trajectory
    
prot_sequence = ['WYSWCSQRLVE']
prot_reducer = ProtRatio(prot_sequence,"/Users/kosio/Repos/PolG_project/Protein_predictions/")
prot_reducer.rationalize()
            

    
        
    


# def position_frequency_matrix(X):
#     '''
#     Parameters
#     ----------
#     X : A list of amino-acid sequences

#     Returns
#     -------
#     A position frequency matrix for all amino acids

#     '''
    