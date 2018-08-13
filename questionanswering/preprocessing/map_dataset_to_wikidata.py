import json
import sys

import click
import tqdm

from questionanswering import config_utils

from wikidata import queries, endpoint_access

@click.command()
@click.argument('config_file_path', default="default_config.yaml")
def process(config_file_path):
    config, logger = config_utils.load_config(config_file_path)

    with open(config['generation']['questions']) as f:
        questions_dataset = json.load(f)

    mapped_dataset = []
    for q in tqdm.tqdm(questions_dataset['Questions'], ascii=True, ncols=100):
        mq = {
            "utterance": q['RawQuestion'],
            "answers": [],
            "answers_str": [],
            "questionid": q['QuestionId']
        }
        for p in q['Parses']:
            mq['answers_str'].extend([a['EntityName'] if a['EntityName'] else a['AnswerArgument'] for a in p['Answers']])
            mq['answers'].extend([a['AnswerArgument'] for a in p['Answers']])

        mq['answers'] = [queries.map_f_id(a) if a.startswith('m') else a for a in mq['answers']]
        for i, a in enumerate(mq['answers']):
            if not a:
                entities = endpoint_access.query_wikidata(queries.query_get_entity_by_label(mq['answers_str'][i]))
                if len(entities) == 1:
                    mq['answers'][i] = entities[0][queries.ENTITY_VAR[1:]]

        mapped_dataset.append(mq)

    with open(config['generation']['save.silver.to'], "w") as out:
        json.dump(mapped_dataset, out, indent=4, sort_keys=True)

    print(f"Coverage: {sum(all(q['answers']) for q in mapped_dataset) / len(mapped_dataset)}")


if __name__ == "__main__":
    process()
