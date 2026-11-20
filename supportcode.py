
import matplotlib.pyplot as plt
def plot_model_accuracies(vgg_acc, densenet_acc):
    models = ['VGG19', 'DenseNet121']
    accuracies = [vgg_acc * 100, densenet_acc * 100]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])

    plt.ylim(0, 100)
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height - 5, f'{height:.2f}%', 
                 ha='center', va='bottom', fontsize=12, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/accuracy_plot.png')
    plt.close()

if __name__ == '__main__':
    plot_model_accuracies(vgg_accuracy, densenet_accuracy)
