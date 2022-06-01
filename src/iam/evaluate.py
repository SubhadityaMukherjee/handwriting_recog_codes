import tensorflow as tf
from models import *
from utilsiam import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--steps', type=int, default=200)

    args = parser.parse_args()
    dataset_path = args.dataset

    char_table_path = os.path.join(dataset_path, "character_table.txt")

    char_table = CharTable(char_table_path)
    steps = args.steps
    image_height = get_meta_info(path=os.path.join(dataset_path, "train"))["average_height"]
    model = CtcModel(
        units=80, num_labels=char_table.size, height=image_height, channels=1
    )

    model.load("conv_lstm_model")
    print("loaded model")

    lines_generator = LinesGenerator(dataset_path+"test", char_table=char_table, batch_size=1, batch_adapter=CTCAdapter())
    evaluator = LEREvaluator(model, lines_generator, steps=steps, char_table=char_table)
    cer = evaluator.evaluate()
    print('Average CER metric without spellcheck is {}'.format(cer))

    evaluator = LEREvaluator(model, lines_generator, steps=steps,char_table=char_table, spellche=True)
    cer = evaluator.evaluate()
    print('Average CER metric with spellcheck is {}'.format(cer))


    