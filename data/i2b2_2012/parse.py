import os
import pandas as pd
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

def index_sentences(report):
	sentence_dict = {}
	sentences = report.split('\n')
	for sentence in sentences:
		sentence_dict[sentence] = {'start': report.index(sentence),
		'end': report.index(sentence)+len(sentence)}
	return sentence_dict

def parse_events(events, sentence_start_end_dict, file):
	temp_df = pd.DataFrame(columns=['sentence_id', 'sentence', 'concept', 
		'concept_start', 'concept_end', 'label'])

	sentence, concept_start, concept_end = None, None, None
	for event in events:
		if event.getAttribute('type')=='PROBLEM':
			concept = event.getAttribute('text')
			concept = concept.replace("'", '&apos;')
			concept = concept.replace('"', '&quot;')
			concept = concept.replace('&apos;', 'temp-replace-apos')
			concept = concept.replace('&quot;', 'temp-replace-quot')
			concept = concept.replace('&', '&amp;')
			concept = concept.replace('&amp;amp;', '&amp;')

			concept = concept.replace('temp-replace-apos', '&apos;')
			concept = concept.replace('temp-replace-quot', '&quot;')

			concept_start_in_sent, concept_end_in_sent = int(event.getAttribute('start')), int(event.getAttribute('end'))

			for key, value in sentence_start_end_dict.items():
				if concept_start_in_sent >= value['start']-20:
					if concept in key:
						start = str(key).index(concept)
						concept_start, concept_end = start, start+len(concept) 
						sentence = key
						break
					else:
						continue

			if sentence == None:
				print(concept)

			sentence_id = str(file.split('.')[0]) + '_' + event.getAttribute('id')
			label = None

			if event.getAttribute('modality')=='FACTUAL':
				label = 'P' if event.getAttribute('polarity')=='POS' else 'N'
			elif event.getAttribute('modality')=='CONDITIONAL':
				label = 'C'
			elif event.getAttribute('modality')=='POSSIBLE':
				label = 'U'
			elif event.getAttribute('modality')=='HYPOTHETICAL':
				label = 'H'
			else:
				print(event.getAttribute('modality'))

			temp_df = temp_df.append({'sentence_id': sentence_id, 
				'sentence': sentence, 
				'concept': concept, 
				'concept_start': concept_start, 
				'concept_end': concept_end, 
				'label': label}, ignore_index=True)
	return temp_df 

def parse_xml_to_csv(dataset):
	dataset_dir = {'train': './training_cleaned/', 'test': './test_cleaned/'}

	xml_dir = dataset_dir[dataset]
	df = pd.DataFrame(columns=['sentence_id', 'sentence', 'concept', 
		'concept_start', 'concept_end', 'label'])

	files = [file for file in os.listdir(xml_dir) if file.endswith('.xml')]
	for file in files:
		xml_file = xml_dir + file
		try:
			tree = ET.parse(xml_file)
		except:
			print(xml_file)

		root = tree.getroot()
		doc = minidom.parse(xml_file)   
		events = doc.getElementsByTagName('EVENT')

		for child in root:
			if child.tag == 'TEXT':
				report = child.text
				break

		sentence_start_end_dict = index_sentences(report)
		temp_df = parse_events(events, sentence_start_end_dict, file)
		df = df.append(temp_df, ignore_index=True)

	df = df.sort_values(by=['sentence_id'])
	df.to_csv(dataset+'.csv', index=False)

if __name__ == '__main__':
	parse_xml_to_csv('train')
	parse_xml_to_csv('test')