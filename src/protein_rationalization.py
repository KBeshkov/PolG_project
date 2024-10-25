#Protein optimization playground
import rmsd
import os
import itertools
import numpy as np
import subprocess
from Bio.PDB import PDBParser, Superimposer
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.Align import substitution_matrices
from tqdm import tqdm
import torch
import esm
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
parser = PDBParser(QUIET=True)

class ProtRatio:
    def __init__(self,sequence,save_path, opt_algorithm, strategy='alphafold', size_penalty = 0.005):
        self.sequence = self.to_elements(sequence)
        self.protein_index = 0
        self.save_path = save_path
        self.opt_algorithm = opt_algorithm
        self.strategy = strategy
        #self.structure_original = parser.get_structure('PolG_structure', '../Structures/pdb8udk.ent')#self.structure_prediciton(self.sequence)
        self.size_penalty = size_penalty
        if strategy=='alphafold':
            self.structure_original = self.structure_prediciton(self.sequence)
            self.atoms = self.protein_coordinates(self.structure_original)
        elif strategy=='esm-embeddings':
            self.embedding_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.embedding_model.eval()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.structure_original = self.structure_prediciton(np.array(self.sequence)[:,None])

    def to_elements(self, sequence):
        return [char for char in sequence[0]]

    def to_list(self, sequence):
        return ["".join(sequence)]

    def protein_coordinates(self, structure):
        chain = next(structure.get_chains())
        coords, seq = get_residue_data(chain)
        return coords

    def esm_embedding(self,sequence):
        sequences = [("protein"+str(i), self.to_list(sequence[:,i])[0]) for i in range(len(sequence.T))]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)

        with torch.no_grad():
            results = self.embedding_model(batch_tokens, repr_layers=[self.embedding_model.num_layers], return_contacts=False)
            token_embeddings = results["representations"][self.embedding_model.num_layers]
        embeddings = []
        for i, (_, seq) in enumerate(sequences):
            emb = token_embeddings[i, 1 : len(seq) + 1].mean(0) #think about this averaging thing
            embeddings.append(emb)
        return torch.vstack(embeddings)

    def structure_prediciton(self, protein):
        if self.strategy=='alphafold':
            self.save_fasta(*self.to_list(protein), self.save_path+str(self.protein_index)+'.fasta')
            subprocess.run(['colabfold_batch', '--num-models=1',self.save_path+str(self.protein_index)+'.fasta', self.save_path+'out'+str(self.protein_index)])
            for file_name in os.listdir(self.save_path+'out'+str(self.protein_index)):
                if file_name.startswith('seq'+str(self.protein_index)+'_unrelaxed_rank_001'):
                    matching_filename = file_name
            struct_file = parser.get_structure("protein_structure", self.save_path+'out'+str(self.protein_index)+'/'+matching_filename)
            return struct_file
        elif self.strategy=='esm-embeddings':
            embedding = self.esm_embedding(protein)
            return embedding
    def distance_function(self,x, seq):
        if self.strategy=='alphafold':
            atoms_x = self.protein_coordinates(x)
            res = tm_align(self.atoms, atoms_x, self.to_list(self.sequence)[0], self.to_list(seq)[0])
            return res.rmsd + self.size_penalty*len(seq)
        elif self.strategy=='esm-embeddings':
            return torch.cdist(x,self.structure_original), self.size_penalty*(len(self.sequence)-len(seq))

    def save_fasta(self, sequence, name):
        with open(name, "w") as fasta_file:
            fasta_file.write(">seq"+str(self.protein_index)+f"\n{sequence}")

    def rationalize(self, n_steps = 3):
        new_protein = self.sequence.copy()
        protein_trajectory = [self.to_list(new_protein)]
        protein_scores = np.zeros([n_steps,self.opt_algorithm.population_size])
        protein_len_scores = []
        for n in tqdm(range(n_steps)):
            self.protein_index += 1
            new_protein = self.opt_algorithm.step(new_protein)
            struct = self.structure_prediciton(new_protein)
            score, len_score = self.distance_function(struct, new_protein)
            protein_trajectory.append([self.to_list(new_protein[:,i]) for i in range(len(new_protein.T))])
            protein_scores[n] = score.detach().flatten()
            protein_len_scores.append(len_score)
            new_protein = new_protein[:,torch.argmin(score)]
        return protein_trajectory, protein_scores, protein_len_scores



class EvolutionAlgorithm:
    def __init__(self, mutation_rate, crossover_rate, n_deletions, population_size = 2, generations = 1, msa = False):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_deletions = n_deletions
        self.population_size = population_size
        self.generations = generations
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.msa = msa
        self.probability_mask = np.tile(self.msa_to_probability(),[self.population_size,1]).T
        if self.msa !=False:
            self.background_rates = self.get_aa_background_rates()
            self.substitution_matrix = substitution_matrices.load("PAM250")
            self.probability_matrix = self.get_aa_probability()

    def sequences_to_numeric(self,sequence):
        amino_acid_mapping = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
            'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': -2, '-': -1}

        numerical_sequences = []

        for record in sequence:
            sequence = str(record.seq)  # Get the sequence as a string
            numerical_sequence = [amino_acid_mapping[aa] for aa in sequence]
            numerical_sequences.append(numerical_sequence)

        numerical_array = np.array(numerical_sequences)
        return numerical_array

    def msa_to_probability(self):
        numerical_array = self.sequences_to_numeric(self.msa)
        preserved_regions = np.logical_and(numerical_array==numerical_array[306],(numerical_array!=-1)*(numerical_array!=-2))
        probability_mask = np.sum(preserved_regions,0)/len(preserved_regions)
        probability_mask = 1-probability_mask[probability_mask!=0]
        norm_prob = probability_mask/np.linalg.norm(probability_mask,1)
        return norm_prob

    def get_aa_background_rates(self):
        acids, counts = np.unique(self.msa,return_counts=True)
        acid_mask = np.where(np.logical_and(acids!='-',acids!='X'))
        background_prob = counts[acid_mask]/sum(counts[acid_mask])
        return background_prob

    def get_aa_probability(self):
        sorted_amino_acids = list(self.amino_acids)
        sorted_sub_matrix = np.zeros((len(self.amino_acids), len(self.amino_acids)))

        # Fill the sorted matrix based on sorted amino acid order
        for i, aa1 in enumerate(sorted_amino_acids):
            for j, aa2 in enumerate(sorted_amino_acids):
                sorted_sub_matrix[i, j] = self.substitution_matrix[aa1, aa2]
        probability_matrix = self.background_rates[:,np.newaxis] * (10 ** (sorted_sub_matrix / 10))
        probability_matrix = probability_matrix/probability_matrix.sum(axis=1, keepdims=True)

        return probability_matrix


    def mutation(self, sequence):
        if len(np.array(sequence).shape)!=2:
            new_sequence = np.tile(sequence,[self.population_size,1]).T
        else:
            new_sequence = np.copy(sequence)
        if self.msa != False:
            mutation_mask = np.zeros((len(sequence),self.population_size)).astype(int)
            rand_mutations = np.copy(new_sequence)
            for i in range(self.population_size):
                n_mutations = sum(np.random.rand(len(sequence))<self.mutation_rate)
                mutation_locations = np.random.choice(np.arange(0,len(sequence)),p = self.probability_mask[:,i],size=n_mutations,replace=False)
                mutation_mask[mutation_locations,i] = 1
                for mut in np.where(mutation_mask[:,i]==1)[0]:
                    current_acid = np.where(new_sequence[mut,i] == np.array(list(self.amino_acids)))[0][0]
                    rand_mutations[mut,i] = np.random.choice(list(self.amino_acids), size=1, p=self.probability_matrix[current_acid])[0]

        else:
            mutation_mask = np.random.rand(len(sequence),self.population_size)<self.mutation_rate
            rand_mutations = np.random.choice(list(self.amino_acids),size=mutation_mask.shape)
        new_sequence[mutation_mask] = rand_mutations[mutation_mask]
        return new_sequence

    def deletion(self, sequence):
        if self.msa != False:
            deletion_mask = np.zeros((self.n_deletions,self.population_size)).astype(int)
            for i in range(self.population_size):
                deletion_mask[:,i] = np.random.choice(np.arange(0,len(sequence)),p = self.probability_mask[:,i],size=(self.n_deletions),replace=False)
        else:
            deletion_mask = np.random.choice(np.arange(0,len(sequence)), size=(self.n_deletions,self.population_size),replace=False)
        new_sequence = np.vstack([np.delete(sequence[:,i],deletion_mask[:,i]) for i in range(self.population_size)]).T
        self.probability_mask = np.vstack([np.delete(self.probability_mask[:,i],deletion_mask[:,i]) for i in range(self.population_size)]).T
        self.probability_mask = self.probability_mask/np.linalg.norm(self.probability_mask,1,axis=0)
        return new_sequence

    def crossover(self,sequence):
        mutated_sequence = self.mutation(sequence)
        new_sequence = np.copy(sequence)
        for i in range(self.population_size):
            cross_point = np.random.randint(1, len(sequence)-2)
            new_sequence[:,i] = list(sequence[:cross_point,i]) + list(mutated_sequence[cross_point:,i])
        return new_sequence

    def step(self, sequence):
        new_seq = list(sequence)
        for gen in range(self.generations):
            new_seq = self.mutation(new_seq)
            new_seq = self.deletion(new_seq)
            if self.crossover_rate>np.random.rand():
                new_seq = self.crossover(new_seq)
        return new_seq
        
        
from Bio import AlignIO

# Define the path to your .aln file
aln_file = "/home/users/constb/Code/PolG_project/data/POLG_MSA_muscle.aln"

# Read the alignment using AlignIO
alignment = AlignIO.read(aln_file, "fasta")

size_penalty = 0.05
PolG = [list("MSRLLWRKVAGATVGPGPVPAPGRWVSSSVPASDPSDGQRRRQQQQQQQQQQQQQPQQPQVLSSEGGQLRHNPLDIQMLSRGLHEQIFGQGGEMPGEAAVRRSVEHLQKHGLWGQPAVPLPDVELRLPPLYGDNLDQHFRLLAQKQSLPYLEAANLLLQAQLPPKPPAWAWAEGWTRYGPEGEAVPVAIPEERALVFDVEVCLAEGTCPTLAVAISPSAWYSWCSQRLVEERYSWTSQLSPADLIPLEVPTGASSPTQRDWQEQLVVGHNVSFDRAHIREQYLIQGSRMRFLDTMSMHMAISGLSSFQRSLWIAAKQGKHKVQPPTKQGQKSQRKARRGPAISSWDWLDISSVNSLAEVHRLYVGGPPLEKEPRELFVKGTMKDIRENFQDLMQYCAQDVWATHEVFQQQLPLFLERCPHPVTLAGMLEMGVSYLPVNQNWERYLAEAQGTYEELQREMKKSLMDLANDACQLLSGERYKEDPWLWDLEWDLQEFKQKKAKKVKKEPATASKLPIEGAGAPGDPMDQEDLGPCSEEEEFQQDVMARACLQKLKGTTELLPKRPQHLPGHPGWYRKLCPRLDDPAWTPGPSLLSLQMRVTPKLMALTWDGFPLHYSERHGWGYLVPGRRDNLAKLPTGTTLESAGVVCPYRAIESLYRKHCLEQGKQQLMPQEAGLAEEFLLTDNSAIWQTVEELDYLEVEAEAKMENLRAAVPGQPLALTARGGPKDTQPSYHHGNGPYNDVDIPGCWFFKLPHKDGNSCNVGSPFAKDFLPKMEDGTLQAGPGGASGPRALEINKMISFWRNAHKRISSQMVVWLPRSALPRAVIRHPDYDEEGLYGAILPQVVTAGTITRRAVEPTWLTASNARPDRVGSELKAMVQAPPGYTLVGADVDSQELWIAAVLGDAHFAGMHGCTAFGWMTLQGRKSRGTDLHSKTATTVGISREHAKIFNYGRIYGAGQPFAERLLMQFNHRLTQQEAAEKAQQMYAATKGLRWYRLSDEGEWLVRELNLPVDRTEGGWISLQDLRKVQRETARKSQWKKWEVVAERAWKGGTESEMFNKLESIATSDIPRTPVLGCCISRALEPSAVQEEFMTSRVNWVVQSSAVDYLHLMLVAMKWLFEEFAIDGRFCISIHDEVRYLVREEDRYRAALALQITNLLTRCMFAYKLGLNDLPQSVAFFSAVDIDRCLRKEVTMDCKTPSNPTGMERRYGIPQGEALDIYQIIELTKGSLEKRSQPGP")]
Evol = EvolutionAlgorithm(0.01,0.001,1,generations=1,population_size=10,msa=alignment)
prot_reducer = ProtRatio(PolG,"/Code/PolG_project/Protein_predictions/",Evol, strategy='esm-embeddings', size_penalty=size_penalty)
new_prot = prot_reducer.rationalize(200)
#new_prot = Evol.step(PolG, generations=350)

np.save("/home/users/constb/Code/PolG_project/data/protein_scores.npy",new_prot[1])
import pickle
with open("/home/users/constb/Code/PolG_project/data/protein_sequences", "wb") as fp:
   pickle.dump(new_prot[0], fp)
