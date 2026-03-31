# Modyfikacje pracy — wersja z Carry (punkty 1-4)

Poniżej znajdują się teksty do wstawienia/zastąpienia w odpowiednich sekcjach pracy. Fragmenty oznaczone [NOWY] to nowe akapity do dodania, [ZMIANA] to modyfikacje istniejących akapitów.

---

## 1. Metodologia → "Podejście sekwencyjne: Football as Language Modeling"

### [ZMIANA] Pierwszy akapit — rozszerzyć listę zdarzeń o Carry:

Kluczową intuicją proponowanego podejścia jest potraktowanie meczu piłkarskiego jako problemu sekwencyjnego, w którym kolejne zdarzenia (podania, prowadzenia piłki, strzały, gole) tworzą uporządkowaną sekwencję zależną od kontekstu chronologicznego - podobnie jak słowa w zdaniu tworzą tekst o spójnym znaczeniu. W tej analogii:

- pojedyncze zdarzenie (Pass, Carry, Shot) odpowiada tokenowi w sekwencji
- akcja ofensywna (łańcuch posiadania) odpowiada zdaniu
- pozycja na boisku stanowi kontekst każdego tokenu
- końcowy rezultat (GOAL/NO_GOAL) to etykieta klasyfikacyjna dla całej sekwencji

### [NOWY] Akapit do dodania po definicji łańcucha posiadania (po "NO_GOAL (wszystkie pozostałe przypadki)"):

Istotnym rozszerzeniem modelu względem wcześniejszych podejść jest włączenie prowadzenia piłki (Carry) jako pełnoprawnego typu zdarzenia w sekwencji. W standardowej analityce piłkarskiej zdarzenia typu Carry - definiowane jako przemieszczenie zawodnika z piłką między innymi zdarzeniami (podaniem, strzałem lub utratą piłki) - są często pomijane lub traktowane jako artefakt danych. Tymczasem prowadzenie piłki w strefę zagrożenia stanowi fundamentalny element budowania akcji ofensywnej. Zawodnik penetrujący dryblingiem pole karne generuje realne zagrożenie bramkowe, które powinno być uwzględnione w modelu Expected Threat. Włączenie Carry do słownika tokenów pozwala modelowi uchwycić pełniejszy obraz dynamiki posiadania piłki, gdzie sekwencja Pass → Carry → Pass → Shot lepiej oddaje rzeczywisty przebieg akcji niż uproszczona wersja Pass → Pass → Shot.

### [ZMIANA] Ostatni akapit — dodać wzmiankę o Carry:

Ta analogia nie jest jedynie metaforyczna - pozwala bezpośrednio wykorzystać architektury neuronowe zaprojektowane pierwotnie dla przetwarzania języka naturalnego, w szczególności Transformer, który dzięki mechanizmowi uwagi (attention mechanism) potrafi modelować złożone zależności między elementami sekwencji niezależnie od ich odległości czasowej. Kluczową przewagą sekwencyjnego podejścia jest zdolność do uczenia się dependencies między akcjami: model może nauczyć się, że seria krótkich podań w środku pola często poprzedza agresywne podanie w głąb, że prowadzenie piłki w stronę pola karnego znacząco zwiększa prawdopodobieństwo strzału, lub że pozycja drugiego podania w akcji silnie zależy od kierunku pierwszego. Tego typu zależności są niedostępne dla modeli operujących na izolowanych akcjach.

---

## 2. Metodologia → sekcja MDN (2.4)

### [NOWY] Akapit do dodania na końcu sekcji o MDN — opis architektury Dual MDN:

Specyfika modelowania sekwencji piłkarskich zawierających zarówno podania jak i prowadzenia piłki ujawnia istotny problem w zastosowaniu pojedynczej sieci MDN do predykcji pozycji. Rozkłady przestrzenne podań i prowadzeń piłki różnią się fundamentalnie: podania charakteryzują się dużą różnorodnością kierunków i dystansów - od krótkich podań wstecz po długie przerzuty na drugą stronę boiska - co wymaga większej liczby komponentów Gaussowskich do adekwatnego modelowania wielomodalnego rozkładu. Natomiast prowadzenia piłki są z natury bardziej ograniczone przestrzennie - zawodnik przemieszcza się na stosunkowo krótkie dystanse, a pozycja końcowa jest silnie skorelowana z pozycją początkową.

W celu uwzględnienia tej różnicy zaproponowano architekturę Dual MDN, w której model wykorzystuje dwie niezależne głowice MDN dzielące wspólny backbone Transformera: głowicę pass z 5 komponentami Gaussowskimi oraz głowicę carry z 3 komponentami. Routing między głowicami odbywa się na podstawie typu zdarzenia - podczas treningu typ docelowego zdarzenia determinuje, która głowica generuje predykcję pozycji i otrzymuje gradient, natomiast podczas symulacji Monte Carlo typ próbkowany z rozkładu kategorycznego wybiera odpowiednią głowicę do generowania współrzędnych następnej akcji. Taka architektura pozwala każdej głowicy wyspecjalizować się w modelowaniu rozkładu przestrzennego właściwego dla danego typu zdarzenia, przy znikomym wzroście liczby parametrów modelu (~12 tysięcy dodatkowych wag, co stanowi mniej niż 0.04% całkowitej liczby parametrów).

---

## 3. Metodologia → "Transformer Architecture"

### [ZMIANA] Akapit o pozycji sekwencyjnej i embeddingach — zaktualizować liczbę tokenów i dodać Carry:

Kluczowym aspektem działania Transformera jest sposób, w jaki architektura wykorzystuje informację o pozycji elementów w sekwencji. W oryginalnej implementacji zastosowano sinusoidalne kodowanie pozycji sekwencyjnej, dodając do reprezentacji każdego tokenu wektor zależny od jego pozycji w sekwencji. W prezentowanym modelu Expected Threat, zamiast kodowania pozycji sekwencyjnej, wykorzystujemy opisane wcześniej kodowanie Fouriera dla współrzędnych przestrzennych (rozdział 2.3). Reprezentacja każdego zdarzenia jest zatem sumą dwóch komponentów: uczalnego embeddingu typu akcji (Pass, Carry, Shot, GOAL, NO_GOAL, <pad> — łącznie 6 tokenów) oraz kodowania Fouriera dla maksymalnie czterech współrzędnych przestrzennych (pozycja początkowa i końcowa). Ta połączona reprezentacja jest następnie wprowadzana do warstw Transformer encoder, gdzie mechanizm self-attention może nauczyć się zależności zarówno między typami akcji, jak i ich przestrzennym rozmieszczeniem na boisku. Dzięki temu model może rozpoznawać wzorce takie jak "seria krótkich podań w centralnej strefie", "prowadzenie piłki w kierunku pola karnego zakończone podaniem prostopadłym" czy "przełączenie gry z lewej na prawą stronę", gdzie zarówno sekwencja typów akcji jak i ich trajektorie przestrzenne są istotne dla przewidywania prawdopodobieństwa gola.

---

## 4. Metodologia → "Model Workflow"

### [ZMIANA] Akapit o przepływie danych — zaktualizować o Carry i Dual MDN:

Przepływ danych przez model rozpoczyna się od sekwencji wejściowej składającej się z typów zdarzeń (Pass, Carry, Shot) oraz ich współrzędnych przestrzennych znormalizowanych do przedziału [0,1]. Każde zdarzenie jest reprezentowane jako suma dwóch embeddingów: uczalnego type embedding (6 tokenów → 512 wymiarów) oraz kodowania Fouriera dla czterech współrzędnych (pozycje początkowe i końcowe). W przypadku podań i prowadzeń piłki dostępne są wszystkie cztery współrzędne (punkt startu i punkt docelowy), natomiast strzały posiadają jedynie współrzędne pozycji strzelca, a tokeny terminalne (GOAL, NO_GOAL) nie zawierają informacji przestrzennej. Wynikowa sekwencja 512-wymiarowych wektorów jest przetwarzana przez Transformer encoder z causal masking, który generuje kontekstualne reprezentacje uwzględniające pełną historię poprzedzających zdarzeń. Warstwa wyjściowa type head przewiduje typ następnego zdarzenia, natomiast dwie niezależne głowice MDN — pass head (5 komponentów Gaussowskich) oraz carry head (3 komponenty) — przewidują rozkład przestrzenny pozycji docelowej, przy czym wybór głowicy jest determinowany przez typ przewidywanego zdarzenia.

### [ZMIANA] Akapit o Monte Carlo — dodać informację o routingu Dual MDN:

Wartość Expected Threat dla danej akcji jest obliczana poprzez symulację Monte Carlo: model generuje wiele niezależnych trajektorii kontynuacji akcji poprzez autoregresywne próbkowanie z przewidywanych rozkładów. W każdym kroku symulacji typ następnego zdarzenia jest próbkowany z rozkładu kategorycznego, a następnie — w zależności od wylosowanego typu — odpowiednia głowica MDN generuje współrzędne pozycji: głowica pass dla podań, głowica carry dla prowadzeń piłki. Dla strzałów pozycja końcowa nie jest generowana (strzał nie posiada lokalizacji docelowej). Wartość xT jest estymowana jako udział trajektorii kończących się golem. Aby uniknąć data leakage, sekwencja wejściowa do symulacji Monte Carlo jest konstruowana w następujący sposób: dla akcji krótkich (1-2 eventy w input) wykorzystywana jest cała dostępna informacja, natomiast dla akcji dłuższych brane są pierwsze 3 tokeny z input sequence. Kluczowe jest, że ze względu na causal shift w konstrukcji datasetu, model nigdy nie obserwuje wyniku akcji (GOAL/NO_GOAL) podczas predykcji - te tokeny znajdują się wyłącznie w sekwencji target, co zapewnia, że model przewiduje prawdopodobieństwo gola wyłącznie na podstawie obserwowanych eventów, bez dostępu do informacji o faktycznym wyniku akcji. To podejście naturalnie uwzględnia niepewność modelu i umożliwia obliczanie wartości xT dla dowolnej pozycji w przestrzeni ciągłej, w przeciwieństwie do tradycyjnych metod wymagających dyskretnej siatki pozycji.

---

## 5. Eksperymenty → "Przygotowanie danych"

### [NOWY] Akapit do dodania po opisie filtrowania Pass i Shot — opis włączenia Carry:

Rozszerzenie modelu o zdarzenia typu Carry wymagało dodatkowego etapu przetwarzania danych. Ze zbioru surowych eventów wyodrębniono prowadzenia piłki, które w danych StatsBomb są rejestrowane automatycznie jako przemieszczenie zawodnika z piłką między kolejnymi zdarzeniami. Surowy zbiór zawierał 1 334 386 zdarzeń typu Carry, jednak ich rozkład dystansów był silnie prawoskośny — mediana wynosiła zaledwie 3.5 jednostki StatsBomb, a 23% prowadzeń miało dystans poniżej 1 jednostki, co odpowiada sytuacjom, w których zawodnik przyjmuje piłkę i praktycznie nie zmienia pozycji. Włączenie wszystkich prowadzeń bez filtrowania wprowadzałoby do modelu znaczną ilość szumu — tokeny Carry o zerowym lub bliskim zeru przemieszczeniu nie niosą informacji przestrzennej istotnej dla predykcji zagrożenia bramkowego.

W celu eliminacji tego szumu zastosowano filtr dystansowy, włączając do sekwencji jedynie prowadzenia piłki o dystansie większym lub równym progowi określonemu w jednostkach StatsBomb (gdzie boisko ma wymiary 120×80). Optymalny próg filtrowania został wybrany eksperymentalnie w ramach procesu optymalizacji opisanego w sekcji 3.2. Każde zdarzenie Carry po filtrowaniu zawiera pełną informację przestrzenną analogiczną do podań — współrzędne początkowe (pozycja zawodnika w momencie rozpoczęcia prowadzenia) oraz współrzędne końcowe (pozycja po zakończeniu prowadzenia), znormalizowane do przedziału [0,1].

### [ZMIANA] Opis struktury eventów — dodać Carry:

Każde zdarzenie w sekwencji jest reprezentowane jako struktura zawierająca typ zdarzenia oraz współrzędne przestrzenne, przy czym konkretna zawartość zależy od typu eventu:

Pass - zawiera pełną informację przestrzenną:
type = 'Pass'
(x, y) - znormalizowane współrzędne początkowe podania [0,1]
(end_x, end_y) - znormalizowane współrzędne końcowe podania [0,1]

Carry - zawiera pełną informację przestrzenną:
type = 'Carry'
(x, y) - znormalizowane współrzędne początkowe prowadzenia [0,1]
(end_x, end_y) - znormalizowane współrzędne końcowe prowadzenia [0,1]

Shot - zawiera jedynie lokalizację strzału:
type = 'Shot'
(x, y) - znormalizowane współrzędne pozycji strzelca [0,1]
(end_x, end_y) - brak (None)

GOAL / NO_GOAL - tokeny terminalne bez współrzędnych:
type = 'GOAL' lub 'NO_GOAL'
(x, y) - brak (None)
(end_x, end_y) - brak (None)

### [ZMIANA] Opis słownika:

Dodatkowo wykorzystano token <pad> służący do wyrównywania długości sekwencji w ramach batchy podczas treningu. Końcowy słownik składa się zatem z 6 tokenów: {Pass: 0, Shot: 1, GOAL: 2, NO_GOAL: 3, <pad>: 4, Carry: 5}.

### [NOWY] Akapit o truncation i context length:

Sekwencje posiadania piłki w surowych danych wykazują znaczne zróżnicowanie długości — od pojedynczych zdarzeń do ponad 100 eventów w najdłuższych akcjach. Włączenie prowadzeń piłki do sekwencji dodatkowo zwiększa ich średnią długość. W celu zapewnienia efektywnego treningu oraz uniknięcia nadmiernego paddingu zastosowano truncation, ograniczając sekwencje do ustalonej maksymalnej liczby zdarzeń. Dla akcji przekraczających ten limit zachowywane są ostatnie zdarzenia w łańcuchu posiadania, co zapewnia, że model zawsze obserwuje końcowy fragment akcji najbardziej istotny dla predykcji wyniku. Optymalny context length — rozumiany jako maksymalna liczba zdarzeń przed tokenem terminalnym — został wybrany eksperymentalnie w ramach procesu optymalizacji opisanego w sekcji 3.2, z uwzględnieniem kompromisu między retencją informacji a efektywnością obliczeniową.
