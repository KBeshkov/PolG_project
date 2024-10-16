#Protein optimization playground
import rmsd
import os
import itertools
import numpy as np
import subprocess
from Bio.PDB import PDBParser, Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
parser = PDBParser(QUIET=True)


class ProtRatio:
    def __init__(self,sequence,save_path, opt_algorithm):
        self.sequence = self.to_elements(sequence)
        self.protein_index = 0
        self.save_path = save_path
        self.opt_algorithm = opt_algorithm
        self.structure_original = parser.get_structure('PolG_structure', '/Users/kosio/Repos/PolG_project/Structures/pdb8udk.ent')#self.structure_prediciton(self.sequence)
        #self.structure_original = self.structure_prediciton(self.sequence)
        self.atoms = self.protein_coordinates(self.structure_original)
        
    def to_elements(self, sequence):
        return [char for char in sequence[0]]
    
    def to_list(self, sequence):
        return ["".join(sequence)]

    def protein_coordinates(self, structure):
        chain = next(structure.get_chains())
        coords, seq = get_residue_data(chain)
        print(coords.shape)
        #coordinates = []
        #for model in structure:
        #    for chain in model:
        #        for residue in chain:
        #            for atom in residue:
        #                coordinates.append(atom.coord)
        return coords
    
    def structure_prediciton(self, protein):
        self.save_fasta(*self.to_list(protein), self.save_path+str(self.protein_index)+'.fasta')
        subprocess.run(['colabfold_batch', self.save_path+str(self.protein_index)+'.fasta', self.save_path+'out'+str(self.protein_index)])
        for file_name in os.listdir(self.save_path+'out'+str(self.protein_index)):
            if file_name.startswith('seq'+str(self.protein_index)+'_unrelaxed_rank_001'):
                matching_filename = file_name
        struct_file = parser.get_structure("protein_structure", self.save_path+'out'+str(self.protein_index)+'/'+matching_filename)
        return struct_file
    
    def distance_function(self,x, seq):
        atoms_x = self.protein_coordinates(x)
        res = tm_align(self.atoms, atoms_x, self.to_list(self.sequence)[0], self.to_list(seq)[0])
        return res.rmsd

    def save_fasta(self, sequence, name):
        with open(name, "w") as fasta_file:
            fasta_file.write(">seq"+str(self.protein_index)+f"\n{sequence}")
    
    def rationalize(self, n_steps = 3):
        new_protein = self.sequence.copy()
        protein_trajectory = [new_protein]
        protein_scores = []
        for n in range(n_steps):
            self.protein_index += 1
            new_protein = self.opt_algorithm.step(new_protein)
            struct = self.structure_prediciton(new_protein)
            protein_trajectory.append(self.to_list(new_protein))
            score = self.distance_function(struct, new_protein)
            protein_scores.append(score)
        return protein_trajectory, protein_scores
    
    
class EvolutionAlgorithm:
    def __init__(self, mutation_rate, crossover_rate, n_deletions, generations = 1):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_deletions = n_deletions
        self.generations = generations
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
    def mutatation(self, sequence):
        mutation_mask = np.random.rand(len(sequence))<self.mutation_rate
        new_sequence = np.array(sequence)
        new_sequence[mutation_mask] = np.random.choice(list(self.amino_acids),size=sum(mutation_mask))
        return list(new_sequence)
    
    def deletion(self, sequence):
        deletion_mask = np.random.choice(np.arange(0,len(sequence)), size=self.n_deletions)
        new_sequence = np.delete(list(sequence), deletion_mask)
        return list(new_sequence)
    
    def crossover(self,sequence1, sequence2):
        cross_point = np.random.randint(1, len(sequence1)-2)
        return sequence1[:cross_point] + sequence2[cross_point:]
    
    def step(self, sequence):
        new_seq = list(sequence)
        for gen in range(self.generations):
            new_seq = self.mutatation(new_seq)
            new_seq = self.deletion(new_seq)
            
            if np.random.rand()<self.crossover_rate:
                new_seq2 = self.deletion(new_seq)
                new_seq2 = self.mutation(new_seq)

                new_seq = self.crossover(new_seq, new_seq2)
        return new_seq
        
        
        
        
        
        
PolG = "MSRLLWRKVAGATVGPGPVPAPGRWVSSSVPASDPSDGQRRRQQQQQQQQQQQQQPQQPQVLSSEGGQLRHNPLDIQMLSRGLHEQIFGQGGEMPGEAAVRRSVEHLQKHGLWGQPAVPLPDVELRLPPLYGDNLDQHFRLLAQKQSLPYLEAANLLLQAQLPPKPPAWAWAEGWTRYGPEGEAVPVAIPEERALVFDVEVCLAEGTCPTLAVAISPSAWYSWCSQRLVEERYSWTSQLSPADLIPLEVPTGASSPTQRDWQEQLVVGHNVSFDRAHIREQYLIQGSRMRFLDTMSMHMAISGLSSFQRSLWIAAKQGKHKVQPPTKQGQKSQRKARRGPAISSWDWLDISSVNSLAEVHRLYVGGPPLEKEPRELFVKGTMKDIRENFQDLMQYCAQDVWATHEVFQQQLPLFLERCPHPVTLAGMLEMGVSYLPVNQNWERYLAEAQGTYEELQREMKKSLMDLANDACQLLSGERYKEDPWLWDLEWDLQEFKQKKAKKVKKEPATASKLPIEGAGAPGDPMDQEDLGPCSEEEEFQQDVMARACLQKLKGTTELLPKRPQHLPGHPGWYRKLCPRLDDPAWTPGPSLLSLQMRVTPKLMALTWDGFPLHYSERHGWGYLVPGRRDNLAKLPTGTTLESAGVVCPYRAIESLYRKHCLEQGKQQLMPQEAGLAEEFLLTDNSAIWQTVEELDYLEVEAEAKMENLRAAVPGQPLALTARGGPKDTQPSYHHGNGPYNDVDIPGCWFFKLPHKDGNSCNVGSPFAKDFLPKMEDGTLQAGPGGASGPRALEINKMISFWRNAHKRISSQMVVWLPRSALPRAVIRHPDYDEEGLYGAILPQVVTAGTITRRAVEPTWLTASNARPDRVGSELKAMVQAPPGYTLVGADVDSQELWIAAVLGDAHFAGMHGCTAFGWMTLQGRKSRGTDLHSKTATTVGISREHAKIFNYGRIYGAGQPFAERLLMQFNHRLTQQEAAEKAQQMYAATKGLRWYRLSDEGEWLVRELNLPVDRTEGGWISLQDLRKVQRETARKSQWKKWEVVAERAWKGGTESEMFNKLESIATSDIPRTPVLGCCISRALEPSAVQEEFMTSRVNWVVQSSAVDYLHLMLVAMKWLFEEFAIDGRFCISIHDEVRYLVREEDRYRAALALQITNLLTRCMFAYKLGLNDLPQSVAFFSAVDIDRCLRKEVTMDCKTPSNPTGMERRYGIPQGEALDIYQIIELTKGSLEKRSQPGP"
Evol = EvolutionAlgorithm(0.01,0.001,1,generations=250)
prot_reducer = ProtRatio(PolG,"./Protein_predictions/",Evol)
new_prot = prot_reducer.rationalize(3)
#new_prot = Evol.step(PolG, generations=350)
print(new_prot)