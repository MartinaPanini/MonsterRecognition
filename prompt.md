# Prima soluzione: modello pre-trainato
voglio scrivere un modello di ML (o deep learning) che riconosca le lattine di Monster Energy in base al loro colore. Il database contiene immagini di tali lattine. Dammi un metodo per estrarre le features per il modello. Dimmi che tipo di modello potrei usare. Spiegami step per step cosa devo fare. Il modello deve riuscire a tracciare il bordo delle singole lattine e in base al loro colore dirmi di che lattina si tratta. 

Visto il database limitato, vorrei utilizzare un modello pre-trainato su un database differente salvato in locale, dove le immagini sono divise in train e test. Scrivi uno script per creare un modello pre-trainato utile al mio scopo. Le immagini sono salvate in formato .jpg, è necessario fare una fase di pre-processing?

# Seconda soluzione: YOLO

voglio scrivere un modello di ML (o deep learning) che riconosca le lattine di Monster Energy in base al loro colore. Il database contiene immagini di tali lattine, strutturato come segue: una cartella per le immagini di train e una per quelle di test, ogni cartella è divisa a sua volta in sottocartelle (il cui nome identifica il nome della lattina, si può considerare come una classe/categoria/label) contenenti le immagini delle lattine relative ad una classe. 

Come prima cosa voglio implementare la detection delle lattine tramite YOLO, in modo da creare un bounding box che contenga una sola lattina. Il box deve avere dimensioni variabili, in modo da adattarsi in base all'immagine. 

Quando estraggo le immagini è necessario eseguire resize e normalizzazione quindi forse è meglio estrarre le immagini dalle cartelle specificate prima, eseguire resize e normalizzazione, e salvare le nuove immagini in un nuovo database. In che modo struttrare il nuovo database decidilo tu. L'importante è che rimanga la corrispondenza tra label ed immagine e che siano ancora divise in train e test