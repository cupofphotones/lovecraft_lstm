import torch
import os
from collections import Counter
import re
import io

lovecraft_folder = 'ds/extra'

class Dataset(torch.utils.data.Dataset):
	def __init__(self,args,):
		self.args = args
		self.words = self.load_words()
		self.uniq_words = self.get_uniq_words()

		self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
		self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

		self.words_indexes = [self.word_to_index[w] for w in self.words]

	def load_words(self):
		text_set = []
		for file in os.listdir(lovecraft_folder):
			f = io.open(os.path.join(lovecraft_folder, file), mode='r', encoding='utf-8')
			text = f.read()
			text_set.append(text.split())

		new_text_set = []
		for text in text_set:
			for word in text:
				word = re.sub('[^АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789]','',word)
				new_text_set.append(word)

		while '' in new_text_set: new_text_set.remove('')
		return new_text_set

	def get_uniq_words(self):
		word_counts = Counter(self.words)
		return sorted(word_counts, key=word_counts.get, reverse=True)

	def __len__(self):
		return len(self.words_indexes) - self.args.sequence_length

	def __getitem__(self, index):
		return(
			torch.tensor(self.words_indexes[index:index+self.args.sequence_length]),
			torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1])
			)