# Contributing

In questo file verrano spiegati gli standard da usare per poter contribuire al progetto

## Table of Contents

- [Contributing](#contributing)
  - [Table of Contents](#table-of-contents)
  - [Github](#github)
    - [Branching](#branching)
    - [Commit](#commit)
      - [Header](#header)
        - [Type](#type)
        - [Scope](#scope)
        - [Description](#description)
      - [Body](#body)
      - [Footer (optional)](#footer-optional)
  - [Python](#python)
    - [Naming](#naming)
    - [SOLID](#solid)
    - [Linter](#linter)
    - [Docs](#docs)

## Github

Il software usato per la gestione del versioning del progetto è github.

### Branching

Per poter lavorare in modo ordinato è necassario l'utilizzo dei branch della repository

La repo è divisa in 3 branch principali:

- `main`: il branch dove risiede il codice in "produzione", quindi la versione "funzionante", anche di prodotti intermedi. In questo branch non sarà possibile effettuare modifiche che non siano prima state testate e che funzionino.
- `dev`: il branch di dev è quello dove si lavora quotidianamente e si pubblicano modifiche al codice. Non è necessario che sia tutto funzionante al 100%, è un branch di sviluppo
- `test`: il branch di test è dove si cerca di verificare se il codice scritto fino a quel punto sia funzionante e robusto.

Per spostarsi da un branch ad un altro basta usare il comando seguente nel terminale nella cartella del progetto.

**NB**: il segno del dollaro è solo un segno usato per indicare che si tratta di un comando del terminale, non è da inserire

```
$ git checkout <nome branch>
```

Per aggiornare il branch locale (che potrebbe rimanere indietro rispetto alla versione attuale presente su github) usare il comando

```
$ git pull origin <nome branch>
```

### Commit

Un messaggio ben strutturato ed esplicativo rende facile la comprensione delle modifiche effettuate. La convenzione da utilizzare per i commit è la [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

L'obiettivo è quello di avere la possibilità di implementare successivamente software automatici per la creazione del CHANGELOG.md in base ai commit eseguiti sulla repository remota.

In breve, un buon messaggio per un commit ha un header, un body e un footer:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

```
fix(type): sistemato tipo dato Image

Aggiunto proprietà "channels" di tipologia integer 32 bit

DEPRECATED: proprietà hva
```

#### Header

```
<type>(<scope>): <description>
```

Esempio

```
fix(type): sistemato tipo dato Image
```

Serve per riassumere il commit, e ha la seguente struttura

##### Type

Il tipo viene scelta dalla seguente lista:
- `fix`: per risoluzione bug
- `feat`: introduzione nuova feature
- `docs`: modifiche ai file di documentazione
- `style`: modifiche di style/design del sistema
- `refactor`: refactor del codice
- `perf`: modifiche per miglioramenti delle prestazioni 
- `test`: modifiche alla suite di test

##### Scope

Lo `scope` dovrebbe essere il nome del componente interessato dalla modifica. I componenti attuali:
- `dataset`: componente per la gestione dei dataset
- `type`: definizione dei tipi di dato per la tipizzazione delle variabili nel codice
- `devtools`: file di utility per il programmatore

##### Description

Descrizione testuale del commit, breve e coincisa.

#### Body

Testo dove si spiegano le motivazioni del commit. Usa verbi al presente indicativo: "riparato", "modificato" ecc...

#### Footer (optional)

Il footer contiene messaggi riguardo `breaking changes` rispetto la versione precedente oppure `deprecated` per funzionalità ormai non supportate. Si possono aggiungere anche riferimenti a issue di Github. Esempio:

```
BREAKING CHANGE: <breaking change summary>

<breaking change description + migration instructions>

Fixes #<issue number>
```

oppure

```
DEPRECATED: <what is deprecated>

<deprecation description + recommended update path>

Closes #<pr number>
```

## Python

Il linguaggio di programmazione principale sarà Python di distribuzione [Anaconda](https://www.anaconda.com/download/) per avere già un ambiente di sviluppo con le principali librerie per lo sviluppo di reti neurali

### Naming

In questa sezione ci sono le regole per il naming di variabili, costanti, funzioni e classi:

- `variabili`: dovrebbero essere in minuscolo, con le parole separate da underscore (es. my_variable).
- `costanti`: di solito sono in maiuscolo, con le parole separate da underscore (es. MY_CONSTANT).
- `funzioni`: dovrebbero essere in minuscolo, con le parole separate da underscore (es. my_function()).
- `classi`: dovrebbero utilizzare la notazione CamelCase, in cui la prima lettera di ogni parola è maiuscola e non ci sono underscore tra le parole (es. MyClass)

### SOLID

Per quanto sia possibile applicare i principi SOLID per la scrittura di un codice pulito, ordinato e facilmente mantenibile.

**NB**: Magari cercare di rispettare questi principi nel refactoring del codice, non perdere troppo tempo all'inizio.

- `Single Responsibility Principle`: ogni classe dovrebbe avere una sola responsabilità o motivo per cambiare.
- `Open/Closed Principle`: le entità software (classi, moduli, funzioni) dovrebbero essere aperte all'estensione, ma chiuse alle modifiche.
- `Liskov Substitution Principle`: gli oggetti di una superclasse dovrebbero essere in grado di essere sostituiti con oggetti di una sottoclasse senza influenzare la correttezza del programma.
- `Interface Segregation Principle`: i clienti non dovrebbero essere costretti a dipendere da interfacce che non utilizzano.
- `Dependency Inversion Principle`: i moduli di alto livello non dovrebbero dipendere da moduli di basso livello. Entrambi dovrebbero dipendere da astrazioni.

### Linter

Un linter è uno strumento di programmazione che analizza il codice sorgente per segnalare errori, bug, costrutti sospetti e problemi di stile, aiutando così a mantenere il codice pulito e aderente alle linee guida di stile

Per vscode si può usare [Pylint](https://marketplace.visualstudio.com/items?itemName=ms-python.pylint)

### Docs

I commenti ben scritti nel codice sono fondamentali per migliorare la leggibilità, facilitare la manutenzione e aiutare altri sviluppatori a capire rapidamente la funzionalità e il comportamento del codice.

Esistono vari plugin per la creazione automatica dei commenti, per vscode si può usare [autoDocstring - Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)