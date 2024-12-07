# Monster Recognition
Seguire un approccio che sfrutti un modello pre-addestrato, come ResNet o MobileNet, per l’estrazione delle feature, integrato con un segmento di rete di segmentazione per il tracciamento dei bordi.

## Step-by-Step

### Pre-processing delle Immagini:

- **Ridimensionamento**: Converti le immagini in una dimensione fissa, come 224x224 o 256x256, per adattarle al modello pre-addestrato.
Normalizzazione: Standardizza i valori di pixel. Per i modelli come ResNet, normalizza l’intervallo dei pixel in base ai valori medi e alla deviazione standard utilizzati in fase di pre-training (per ImageNet, ad esempio).
Data Augmentation: Usa tecniche di data augmentation (flip, rotazione, zoom) per migliorare la robustezza del modello data la limitatezza del dataset.

- **Caricamento del Modello Pre-addestrato**: Carica un modello come ResNet o MobileNet, pre-addestrato su ImageNet. Puoi scegliere di “congelare” i livelli convolutivi (bloccandone i pesi) e aggiungere una testa personalizzata per la classificazione del colore della lattina e una per la segmentazione dei bordi.

- **Feature Extraction e Fine-Tuning**: Usa le feature intermedie estratte dal modello per classificare il colore e segmentare i bordi delle lattine.
- **Classificazione del Colore**: Aggiungi una densa layer sopra il modello pre-addestrato per classificare i colori.
- **Segmentazione dei Bordi**: Aggiungi un piccolo decoder, come una Fully Convolutional Network (FCN) o una U-Net, per generare una maschera dei bordi della lattina.
- **Addestramento del Modello**: Dividi i dati in train e validation set se non l'hai già fatto.
Addestra il modello sui dati utilizzando la cross-entropy per la classificazione e una funzione di loss specifica per la segmentazione, come la binary cross-entropy o la dice loss.
- **Valutazione e Test**: Valuta il modello sul set di test per verificare le prestazioni sia nella classificazione del colore sia nella segmentazione dei bordi.