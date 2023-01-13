
from evaluate import evaluate
from models import transformers
from train import train


def main():
    model = transformers()
    train(model)
    evaluate(model)


main()
