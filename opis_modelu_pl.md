# Opis modelu Expected Threat — dokumentacja techniczna

## 1. Dane wejsciowe i pipeline danych

### 1.1 Zrodlo danych

Dane pochodza z otwartego zbioru StatsBomb (sezony 2015/2016). Kazdy mecz jest opisany sekwencja zdarzen (event stream), gdzie kazde zdarzenie ma typ (Pass, Shot, Carry), pozycje startowa na boisku (x, y) oraz pozycje koncowa (end_x, end_y dla podan i prowadzen pilki). Boisko StatsBomb ma wymiary 120 x 80 jednostek.

### 1.2 Filtrowanie zdarzen

Z pelnego strumienia zdarzen wybieramy tylko trzy typy akcji istotnych z perspektywy budowania zagrozenia: podania (Pass), strzaly (Shot) i prowadzenia pilki (Carry). Strzaly karne sa usuwane, poniewaz nie wynikaja z gry pozycyjnej i zaburzalyby model — ich wartosc zagrozenia jest zdeterminowana regulaminowo, a nie przez kontekst przestrzenny.

### 1.3 Filtrowanie prowadzen pilki (Carry)

Dane StatsBomb rejestruja prowadzenie pilki (Carry) dla kazdego momentu, gdy zawodnik trzyma pilke przy nodze — nawet jesli jest to przesuniecie o kilkadziesiat centymetrow przed oddaniem podania. Takie mikro-prowadzenia sa szumem: nie niosa informacji o budowaniu zagrozenia, a zasmiecaja sekwencje. Model musi odroznic prowadzenie pilki o 30 metrow w kierunku bramki od technicznego przesuniecia o pol metra.

Stosujemy filtr dystansowy: obliczamy odleglosc euklidesowa miedzy pozycja startowa a koncowa prowadzenia (w jednostkach StatsBomb), i zachowujemy tylko prowadzenia dluzsze niz ustalony prog. W pracy testowalismy trzy progi: >= 1 (prawie brak filtrowania), >= 2 (umiarkowany) i >= 3 (restrykcyjny). Prog >= 2 jednostki StatsBomb okazal sie optymalny — usuwa szum bez utraty wartosciowych prowadzen.

### 1.4 Obrot ukladu wspolrzednych

StatsBomb podaje wspolrzedne z perspektywy druzyny wykonujacej akcje. Jesli zdarzenie wykonuje druzyna broniaca (np. przechwyt), jej wspolrzedne sa podane z jej wlasnej perspektywy — inna polowa boiska. Aby zachowac spojnosc przestrzenna w obrebie posiadania, odwracamy wspolrzedne zdarzen druzyny broniacej: x' = 120 - x, y' = 80 - y. Dzieki temu cala sekwencja posiadania jest widziana z perspektywy druzyny atakajacej, a pozycja blizej bramki rywala zawsze odpowiada wyzszym wartosciom x.

### 1.5 Ujednolicenie pozycji koncowej

Rozne typy zdarzen maja pozycje koncowa w roznych kolumnach danych zrodlowych: podania w pass_end_location, prowadzenia w carry_end_location, strzaly nie maja pozycji koncowej. Pipeline ujednolica to do wspolnej kolumny end_location: dla podan bierzemy pass_end_location, dla prowadzen carry_end_location, dla strzalow ustawiamy null. Dzieki temu model widzi kazde zdarzenie jako jednolity wektor [start_x, start_y, end_x, end_y], niezaleznie od typu akcji.

### 1.6 Budowanie sekwencji

Zdarzenia sa grupowane po posiadaniu (match_id + possession). Kazda sekwencja to chronologiczny ciag akcji w ramach jednego posiadania pilki. Na koncu sekwencji dodajemy token terminalny: GOAL jesli posiadanie zakonczylo sie golem, NO_GOAL w przeciwnym przypadku. Token terminalny nie ma wspolrzednych przestrzennych (x, y, end_x, end_y sa ustawione na null). Obecnosc gola jest wykrywana na podstawie atrybutu shot_outcome == 'Goal' w dowolnym zdarzeniu w sekwencji.

### 1.7 Obcinanie sekwencji (truncation)

Posiadania moga miec rozna dlugosc — od jednego podania do kilkudziesieciu akcji. Zbyt dlugie sekwencje sa problematyczne: Transformer ma kwadratowa zlozonosc obliczeniowa wzgledem dlugosci sekwencji, a bardzo dlugie posiadania czesto sa rozproszone i nie niosa wiecej informacji predykcyjnej niz ostatnie kilka-kilkanascie akcji. Dlatego obcinamy sekwencje do ustalonej dlugosci kontekstu (context length), zachowujac ostatnie N zdarzen (te najblizsze zakonczeniu posiadania). Obcinanie dziala od poczatku sekwencji — usuwamy najwczesniejsze zdarzenia, a te blizsze strzalowi/zakonczeniu pozostaja. To decyzja celowa: zdarzenia bezposrednio poprzedzajace strzal lub utrate pilki niosa wiecej informacji predykcyjnej niz poczatek rozegrania od bramkarza.

W pracy testowalismy konteksty od 2 do 23 zdarzen. Optymalny okazal sie kontekst 8 zdarzen — zachowuje okolo 65% pelnych danych, a dalsze wydluzanie kontekstu pogarszalo wyniki (model zaczynal overfittowac na dlugich, czesto szumowych poczatkach posiadania).

Parametr max_seq_len w kodzie to context_length + 2 (miejsce na token terminalny i padding).

### 1.8 Podzial na zbiory (train/val/test)

Sekwencje sa dzielone losowo w proporcji 80/10/10 (train/val/test) z ustalonym seedem (42), co zapewnia reprodukowalnosc. Ten sam seed jest uzywany zarowno w pipeline'ie carry jak i pass-only, co daje powiazane (choc nie identyczne) podzialy — istotne dla uczciwosci porownania miedzy modelami.

### 1.9 Normalizacja wspolrzednych

Wspolrzedne sa normalizowane do zakresu [0, 1]: x = x / 120, y = y / 80. Dzieki temu model operuje na wartosciach znormalizowanych niezaleznie od rozmiaru boiska.

### 1.10 Balansowanie zbioru treningowego

Gole sa rzadkie — tylko okolo 1-2% posiadani konczy sie golem. Gdyby model treniwal na naturalnym rozkladzie, szybko nauczylby sie odpowiadac "NO_GOAL" na wszystko i mialby wysoka dokladnosc, ale zerowa uzytecznosc. Dlatego zbior treningowy jest balansowany do okolo 5% goli — wszystkie posiadania z golem sa zachowane, a posiadania bez gola sa losowo podprobkowane. Zbior walidacyjny pozostaje naturalny (niebalansowany), zeby ocena modelu odzwierciedlala rzeczywiste proporcje.

### 1.11 Slownik tokenow (vocab)

Model operuje na dyskretnych typach zdarzen. Slownik dla modelu z carry zawiera 6 tokenow: Pass (0), Shot (1), GOAL (2), NO_GOAL (3), \<pad\> (4), Carry (5). Model pass-only ma 5 tokenow (bez Carry). Tokeny GOAL i NO_GOAL to tokeny terminalne — model uczy sie je przewidywac. Token \<pad\> sluzy do wyrownywania sekwencji do stalej dlugosci w batchu.

### 1.12 Struktura pojedynczej probki

Kazda probka treningowa to para (input, target) uzyskana przez przesuniecie przyczynowe (causal shift): input to zdarzenia od 1 do N-1, target to zdarzenia od 2 do N. Model widzi sekwencje dotychczasowych akcji i na kazdej pozycji uczy sie przewidywac nastepna akcje. To ten sam mechanizm co w modelach jezykowych GPT — roznica polega na tym, ze zamiast slow mamy zdarzenia pilkarskie, a zamiast jednego tokena mamy typ akcji plus wspolrzedne.

Kazde zdarzenie jest opisane przez:
- **typ** (token ID) — jaki to rodzaj akcji
- **pozycje** — wektor [start_x, start_y, end_x, end_y] — skad i dokad zmierzala akcja
- **maski startowe i koncowe** — dwie osobne maski boolowskie informujace, ktore wspolrzedne sa dostepne:
  - start_mask = True jesli zdarzenie ma wspolrzedne startowe (x, y). Prawdziwa dla Pass, Shot, Carry; falszywa dla tokenow GOAL, NO_GOAL, \<pad\>.
  - end_mask = True jesli zdarzenie ma wspolrzedne koncowe (end_x, end_y). Prawdziwa dla Pass i Carry; falszywa dla Shot (strzal nie ma "pozycji koncowej" — nie rejestrujemy gdzie pilka trafia, bo to juz nie jest budowanie zagrozenia), GOAL, NO_GOAL, \<pad\>.

Sekwencje krotsze niz max_seq_len sa dopelniane paddingiem:
- typy wejsciowe: token \<pad\> (ID 4 lub 5 w zaleznosci od vocab)
- typy docelowe: -100 (specjalna wartosc PyTorcha mowiaca cross-entropy, zeby ignorowala te pozycje)
- pozycje: [0.0, 0.0, 0.0, 0.0]
- maski: False (padding nie wnosi ani informacji przestrzennej, ani nie wchodzi do straty)

### 1.13 Rozdzielenie danych treningowych i ewaluacyjnych

Istotny szczegol implementacyjny: model moze byc trenowany na sekwencjach o innej dlugosci niz te, na ktorych jest ewaluowany. W optymalnej konfiguracji trening odbywa sie na sekwencjach obcietych do kontekstu 8 (max_seq_len = 10), ale ewaluacja xT Monte Carlo uzywa pelnych sekwencji walidacyjnych (eval_max_seq_len = 16, czyli kontekst 14). Motywacja: trening na krotszych sekwencjach zapobiega overfittingowi i jest szybszy, ale ewaluacja na pelnych sekwencjach lepiej odzwierciedla realne warunki uzycia modelu. Zbiory walidacyjne dla treningu (strata epokowa) i dla ewaluacji xT moga pochodzic z roznych plikow danych.

## 2. Architektura modelu

### 2.1 Wejscie modelu — embeddingi

Model przyjmuje dwa rodzaje informacji na kazdej pozycji sekwencji: typ zdarzenia i jego pozycje przestrzenna.

**Embedding typu** to standardowa tablica embeddingow (lookup table) — kazdy z 6 (lub 5) typow zdarzen jest mapowany na wektor o wymiarze d_model = 512.

**Encoding pozycji** uzywa Fourierowskiego kodera pozycji (Fourier Position Encoder). Zamiast podawac surowe wspolrzedne (4 liczby z zakresu [0,1]), kazda wspolrzedna jest rozkladana na 8 czestotliwosci: [1, 2, 4, 8, 16, 32, 64, 128]. Dla kazdej czestotliwosci obliczamy sinus i cosinus: sin(x * freq) i cos(x * freq). To daje 4 wspolrzedne * 8 czestotliwosci * 2 (sin + cos) = 64 cechy Fourierowskie, ktore sa nastepnie rzutowane warstwa liniowa na wymiar d_model = 512.

Motivacja: surowe wspolrzedne [0.5, 0.5] i [0.51, 0.51] sa prawie identyczne numerycznie, ale po rozlozeniu Fourierowskim roznice sa wzmocnione na wysokich czestotliwosciach. Niskie czestotliwosci koduja ogolne polozenie na boisku (polowa wlasna vs polowa rywala), wysokie czestotliwosci — precyzyjna pozycje (np. roznica miedzy polem karnym a linia koncowa). Model uczy sie sam, ktore czestotliwosci sa istotne.

Embedding pozycji jest zerowany maska (start_mask) — pozycje, ktore nie maja wspolrzednych (tokeny terminalne, padding) nie wnosa informacji przestrzennej.

Oba embeddingi sa sumowane (nie konkatenowane): combined = type_embedding + position_embedding.

### 2.2 Transformer Encoder z maska przyczynowa

Sekwencja embeddingow przechodzi przez Transformer Encoder. Kazda warstwa Transformera sklada sie z:
- mechanizmu wieloglowicowej uwagi (multi-head self-attention) z 8 glowicami
- sieci feedforward o wymiarze 2048
- normalizacji warstw (layer normalization) i polaczen rezydualnych (residual connections)
- dropoutu 0.1

Kluczowy element to **maska przyczynowa** (causal mask): gornotrojkatna macierz maskujaca zapewnia, ze kazdy token moze "widziec" tylko tokeny wczesniejsze i siebie samego. Bez tego model moglby "podejrzec" przyszle zdarzenia (w tym token terminalny) i oszukiwac. Maska sprawia, ze model zachowuje sie jak model autoregresyjny — generuje predykcje krok po kroku, tak jak w rzeczywistej grze.

W optymalnej konfiguracji model ma **4 warstwy** Transformera. Testowalismy 4, 8, 12 i 16 warstw — mniejszy model generalizuje lepiej. Przy 16 warstwach model katastrofalnie overfituje (ROC-AUC spada do poziomu losowego ~0.50), przy 8 warstwach wyniki sa gorsze niz przy 4. To sugeruje, ze problem predykcji xT nie wymaga glebokich hierarchii cech — wystarczy kilka warstw uwagi, zeby uchwycic wzorce przestrzenno-sekwencyjne.

### 2.3 Glowice wyjsciowe — co model przewiduje

Z Transformera wychodzi uzyskana reprezentacja hidden state o wymiarze [batch, seq_len, 512]. Na tej reprezentacji operuja dwie (lub trzy) glowice wyjsciowe:

**Glowica typow (type head)** — warstwa liniowa [512 -> vocab_size]. Na kazdej pozycji sekwencji przewiduje rozklad prawdopodobienstwa nad mozliwymi typami nastepnego zdarzenia. "Czy nastepna akcja to podanie, strzal, prowadzenie, gol, czy koniec bez gola?" Trenowana funkcja straty cross-entropy.

**Glowica przestrzenna MDN (Mixture Density Network)** — zamiast przewidywac jedna pozycje nastepnego zdarzenia, model generuje mieszanine rozkladow normalnych (Gaussian Mixture). Kazdy komponent mieszaniny opisuje jedno "skupisko" mozliwych pozycji. Dzieki temu model moze wyrazac niepewnosc ("pilka moze pojsc w lewy rog karne LUB do srodka") zamiast uśredniać do jednego punktu.

### 2.4 Parametry MDN — co dokladnie opisuje kazdy komponent

Kazdy komponent mieszaniny Gaussowskiej jest opisany 8 parametrami (stad mdn_head produkuje n_components * 8 wartosci):

1. **waga komponentu** (weight) — jak prawdopodobny jest ten scenariusz. Przetworzona przez softmax, zeby wagi sumowaly sie do 1.

2-3. **srednia pozycji startowej** (start_mean_x, start_mean_y) — oczekiwana pozycja poczatku nastepnej akcji. Przetworzona przez sigmoid, zeby wynik byl w zakresie [0, 1] (znormalizowane boisko).

4. **odchylenie pozycji startowej** (start_std) — pojedyncze odchylenie wspolne dla obu wspolrzednych startowych. Przetworzone przez exp() i obciete do zakresu [0.005, 0.1]. Dolna granica 0.005 zapobiega kolapsowi rozkladu do delty Diraca (zerowej wariancji), gorna granica 0.1 zapobiega zbyt rozmazanym predykcjom. **Pozycja startowa ma jedno wspolne odchylenie** — zakladamy ze niepewnosc co do pozycji startowej jest izotropowa (taka sama w x i y), bo pozycja startowa to de facto "gdzie jest pilka teraz" i model jest jej stosunkowo pewien.

5-6. **srednia pozycji koncowej** (end_mean_x, end_mean_y) — oczekiwana pozycja zakonczenia nastepnej akcji (np. gdzie trafi podanie). Sigmoid, zakres [0, 1].

7-8. **odchylenia pozycji koncowej** (end_std_x, end_std_y) — **oddzielne odchylenia dla x i y**. Przetworzone przez exp() i obciete do zakresu [0.01, 0.5]. Pozycja koncowa ma oddzielne odchylenia dla kazdej osi, bo niepewnosc co do miejsca docelowego podania moze byc asymetryczna — mozna wiedziec mniej wiecej na jaka wysokosc trafi podanie (y), ale byc mniej pewnym jak daleko doleci (x), albo odwrotnie. Szerszy zakres clampowania ([0.01, 0.5] vs [0.005, 0.1]) odzwierciedla wieksza naturalna niepewnosc co do pozycji koncowej niz startowej.

### 2.5 Forward pass — przebieg obliczen

Pelny przebieg przez model (forward pass) wyglada nastepujaco:

1. **Embedding typow**: tablica lookup mapuje ID typow na wektory [batch, seq_len, 512]
2. **Encoding pozycji**: Fourierowski koder przetwarza wspolrzedne [batch, seq_len, 4] na wektory [batch, seq_len, 512]
3. **Maskowanie pozycji**: embedding pozycji jest mnozony elementowo przez start_mask — pozycje bez wspolrzednych (terminalne, padding) sa zerowane
4. **Sumowanie**: type_embedding + position_embedding = combined [batch, seq_len, 512]
5. **Transformer**: combined przechodzi przez 4 warstwy Transformera z maska przyczynowa, produkujac hidden [batch, seq_len, 512]
6. **Glowica typow**: warstwa liniowa [512 -> vocab_size] na hidden, produkujac logity typow [batch, seq_len, 6]
7. **Glowica MDN pass**: warstwa liniowa [512 -> n_components_pass * 8] na hidden, potem reshape do [batch, seq_len, 5, 8] — piec komponentow, kazdy z 8 parametrami
8. **Glowica MDN carry**: analogicznie [512 -> n_components_carry * 8], reshape do [batch, seq_len, 3, 8]

Reshape glowic MDN jest kluczowy — surowe wyjscie to plaski wektor (np. 40 wartosci dla 5 komponentow * 8 parametrow), ktory musi byc podzielony na logiczna strukture: komponent x parametr. Kazdy komponent dostaje swoje 8 liczb, ktore sa potem parsowane na wage, srednie i odchylenia.

### 2.6 Dual MDN — oddzielne glowice dla podan i prowadzen

Podania i prowadzenia pilki maja fundamentalnie rozne rozklady przestrzenne. Podanie moze poleciec daleko w dowolnym kierunku — krotkie podania do tylu, dlugiego podania przerzuty, centrowania z skrzydla. Prowadzenie pilki to zwykle krotkie przesuniecie w podobnym kierunku, w jakim zawodnik sie poruszal. Zmuszanie jednej glowicy MDN do modelowania obu tych rozkladow jednoczesnie moze prowadzic do rozmycia — model usrednia oba wzorce.

Rozwiazanie: **dwie oddzielne glowice MDN**:
- **mdn_head_pass** — 5 komponentow mieszaniny. Wieksza liczba komponentow pozwala modelowac bogata przestrzen mozliwych podan (krotkie, srednie, dlugie, w roznych kierunkach).
- **mdn_head_carry** — 3 komponenty mieszaniny. Mniejsza liczba, bo prowadzenia sa bardziej ograniczone przestrzennie i nie wymagaja tak bogatego modelowania.

Podczas treningu nastepuje **routing** — strata MDN jest obliczana oddzielnie dla kazdego tokena docelowego. Jesli nastepne zdarzenie to podanie, strata jest liczona z glowicy pass. Jesli to prowadzenie — z glowicy carry. Strzaly uzywaja glowicy pass (maja pozycje startowa, ale nie maja pozycji koncowej). Dzieki temu kazda glowica specjalizuje sie w swoim typie ruchu.

### 2.7 Single MDN — wariant ablacyjny

Jako test ablacyjny sprawdzamy, czy oddzielna glowica carry jest rzeczywiscie potrzebna. W wariancie single MDN istnieje tylko glowica pass (5 komponentow). Dla prowadzen pilki model bierze pierwsze 3 komponenty glowicy pass (slice: mdn_pass[:, :, :3, :]) i uzywa ich zamiast dedykowanej glowicy carry. Oznacza to, ze te 3 komponenty musza jednoczesnie modelowac i podania i prowadzenia — nie maja dedykowanej specjalizacji. Architektura jest poza tym identyczna — ten sam Transformer, ten sam embedding, ta sama glowica typow, ta sama liczba parametrow Transformera. Roznica w liczbie parametrow to tylko brak jednej warstwy liniowej [512 -> 24] (glowicy carry), co stanowi marginalna roznice (~12 500 parametrow z ~25 milionow calego modelu).

## 3. Funkcje straty

### 3.1 Strata typow (type loss)

Standardowa cross-entropy miedzy przewidzianymi logitami typow a rzeczywistymi typami nastepnych zdarzen. Pozycje paddingu (target = -100) sa ignorowane automatycznie. Model moze miec wagi klas (class weights) — np. wyzsze wagi dla strzalow i goli, zeby model poswiecal im wiecej uwagi. W finalnej konfiguracji uzywamy jednolitych wag (uniform weights = 1.0 dla wszystkich typow), poniewaz okazaly sie one optymalnym wyborem.

### 3.2 Strata MDN (MDN loss)

Strata MDN to ujemna logarytmiczna wiarygodnosc (negative log-likelihood) mieszaniny Gaussowskiej. Dla kazdego tokenu obliczamy, jak prawdopodobne sa rzeczywiste wspolrzedne nastepnego zdarzenia wedlug rozkladu przewidzianego przez model.

**NLL pojedynczego rozkładu normalnego** jest obliczana wzorem:

    NLL = 0.5 * (log(2 * pi * sigma^2) + (target - mean)^2 / sigma^2)

Pierwszy skladnik to "kara za niepewnosc" — im szersza dystrybucja (wieksze sigma), tym wieksza strata bazowa. Drugi skladnik to "kara za blad" — im dalej rzeczywista pozycja od sredniej, tym wieksza strata. Model balansuje miedzy precyzja a pewnoscia siebie: zbyt waskie odchylenie daje maly pierwszy skladnik, ale ogromny drugi jesli predykcja jest choc troche nieprecyzyjna.

**Dla kazdego komponentu mieszaniny** NLL jest obliczany oddzielnie dla pozycji startowej (2 wspolrzedne, wspolne odchylenie) i koncowej (2 wspolrzedne, oddzielne odchylenia), a wyniki sa sumowane.

**Laczenie komponentow** odbywa sie przez logsumexp: NLL_mieszaniny = -logsumexp(log(waga_k) - NLL_k). Jest to numerycznie stabilna wersja obliczenia log(sum(waga_k * exp(-NLL_k))). Uzywamy logsumexp zamiast naiwnego log(sum(exp(...))) poniewaz eksponenty moga byc bardzo duze lub bardzo male, co prowadzi do problemow z precyzja numeryczna (overflow/underflow).

**Routing w dual MDN**: strata jest obliczana z obu glow (pass i carry) dla kazdego tokena, ale potem maskowana. Dla tokenow typu Pass i Shot liczy sie NLL z glowicy pass, dla tokenow typu Carry — z glowicy carry. Formalnie: routed_NLL = NLL_pass * (spatial_mask - is_carry) + NLL_carry * is_carry. Spatial_mask zapewnia, ze tylko tokeny z wspolrzednymi wchodza do straty (pomija GOAL, NO_GOAL, \<pad\>). Wynikowa strata jest normalizowana przez liczbe tokenow z wspolrzednymi (spatial_mask.sum()), nie przez calkowita liczbe tokenow — dzieki temu model nie jest "nagradzany" za ignorowanie paddingu.

### 3.3 Interpretacja ujemnej straty

Warto zauwazyc, ze strata MDN (a wiec i laczna strata) moze przyjmowac wartosci ujemne. Ujemna log-likelihood oznacza, ze model przypisuje predykcji prawdopodobienstwo wieksze niz 1 w sensie gestosci — co jest calkowicie normalne dla ciaglych rozkladow prawdopodobienstwa (gestosc moze przekraczac 1, w przeciwienstwie do prawdopodobienstwa). Im bardziej ujemna strata, tym lepiej — model jest pewny swoich predykcji i te predykcje sa trafne.

### 3.4 Laczenie strat

Calkowita strata to suma straty typow i straty MDN: total_loss = loss_ratio * type_loss + mdn_loss. Parametr loss_ratio kontroluje balans — w finalnej konfiguracji ustawiony na 1.0 (rownowaga obu strat). Obie straty sa optymalizowane jednoczesnie — model uczy sie rownoczesnie przewidywac typ nastepnego zdarzenia i jego pozycje przestrzenna. Gradienty obu strat przeplywaja przez wspolny Transformer, co oznacza ze dobre rozumienie pozycji pomaga w predykcji typow i odwrotnie.

## 4. Trening

### 4.1 Optymalizator

AdamW z learning rate 1e-4 i weight decay 0.01. Weight decay zapobiega nadmiernemu wzrostowi wag i dziala jako forma regularyzacji.

### 4.2 Gradient clipping

Gradienty sa obcinane do maksymalnej normy 1.0 (gradient clipping). Zapobiega to eksplozji gradientow, ktora moze wystapic w glebokich sieciach Transformer, szczegolnie we wczesnych fazach treningu.

### 4.3 Wybor najlepszego modelu

Po kazdej epoce obliczamy strate na zbiorze walidacyjnym. Jezeli strata walidacyjna jest nizsza niz najlepsza dotychczasowa, zapisujemy checkpoint modelu (best_model.pt). Po zakonczeniu treningu wczytujemy najlepszy checkpoint do finalnej ewaluacji. Dzieki temu model uzywany do ewaluacji to nie model z ostatniej epoki (czesto juz przeuczony), lecz z epoki o najlepszej generalizacji.

### 4.4 Wagi klas w funkcji straty typow

Funkcja cross-entropy moze przypisywac rozne wagi roznym typom tokenow. Testowalismy konfiguracje z podwyzszonymi wagami dla Shot (10x), GOAL (30x) i NO_GOAL (1.5x) — zeby model poswiecal wiecej uwagi rzadkim, ale kluczowym zdarzeniom. Ostatecznie jednolite wagi (1.0 dla wszystkich typow) okazaly sie optymalnym wyborem — model sam nauczyl sie wlasciwych proporcji bez recznie ustawianych priotytetow.

### 4.5 Parametry treningu

- 15 epok
- batch size 32
- dropout 0.1 (losowe zerowanie 10% polaczen w Transformerze, zapobiega overfittingowi)
- Rozmiar modelu: d_model = 512, 8 glow uwagi, 4 warstwy
- pin_memory = True (dla GPU, przyspiesza transfer danych CPU -> GPU)
- Bez dodatkowych workerow DataLoadera (num_workers = 0) — dane sa na tyle male, ze wielowatkowosc nie pomaga

## 5. Ewaluacja — obliczanie Expected Threat

### 5.1 Idea: Monte Carlo rollouts

Expected Threat (xT) to prawdopodobienstwo, ze posiadanie pilki zakonczy sie golem, biorąc pod uwage biezacy kontekst (pierwsze kilka zdarzen w sekwencji). Nie mozemy tego policzyc analitycznie — model jest zbyt zlozony. Zamiast tego uzywamy symulacji Monte Carlo.

Dla kazdego posiadania w zbiorze walidacyjnym:
1. Bierzemy poczatek sekwencji (pierwsze 3 zdarzenia, lub mniej jesli sekwencja jest krotsza)
2. Uruchamiamy N rownoleglych symulacji (rollouts), kazda zaczynajaca sie od tego samego poczatku
3. W kazdej symulacji model autoregresyjnie generuje nastepne zdarzenia az do tokenu terminalnego (GOAL lub NO_GOAL) lub osiagniecia maksymalnej dlugosci sekwencji
4. xT = odsetek symulacji, ktore zakonczyly sie golem

### 5.2 Generowanie pojedynczego kroku

W kazdym kroku symulacji model:

1. **Przewiduje typ nastepnej akcji** — softmax na logitach typow, potem losowanie z tego rozkladu (multinomial sampling). To wprowadza stochastycznosc: w jednej symulacji nastepna akcja to podanie, w innej strzal. Model sam nauczonego "kiedy strzelac, kiedy podawac".

2. **Sprawdza token terminalny** — jesli wylosowano GOAL, ta symulacja liczy sie jako sukces. Jesli NO_GOAL — jako porazka. Obie symulacje sa deaktywowane (nie generuja dalszych kroków).

3. **Generuje pozycje nastepnej akcji** (jesli akcja jest przestrzenna: Pass, Carry, Shot):
   - Domyslnie pobiera parametry z glowicy pass — parsuje wagi komponentow, srednie i odchylenia z surowego wyjscia MDN
   - Losuje komponent mieszaniny wedlug wag (multinomial sampling) — np. jesli komponent 2 ma wage 0.6, bedzie wybrany w 60% przypadkow
   - Jesli wylasowano Carry i model ma dual MDN: nadpisuje parametry tych rolloutow parametrami z glowicy carry (oddzielne losowanie komponentu z carry head). To oznacza ze w jednym batchu rolloutow czesc moze uzywac glowicy pass, czesc carry — w zaleznosci od wylosowanego typu
   - Losuje pozycje startowa: start_xy = start_mean + randn * start_std, gdzie randn to losowy szum z rokladu normalnego N(0,1). Wspolne odchylenie start_std jest rozszerzane na obie wspolrzedne (izotropowy szum)
   - Losuje pozycje koncowa: end_xy = end_mean + randn * end_std, z oddzielnymi odchyleniami dla x i y (anizotropowy szum)
   - Obcina obie pozycje do zakresu [0, 1] (clamp) — zeby symulacja nie "wyszla poza boisko". Bez tego kroku losowanie z rozkładu normalnego mogloby generowac ujemne wspolrzedne lub wartosci powyzej 1
   - Dla strzalow pozycja koncowa jest ustawiana na [0, 0] — strzaly nie maja end_location w danych, wiec model nie powinien na niej polegac

4. **Dokleja nowe zdarzenie** do sekwencji — nowy typ, nowe wspolrzedne i maska (True, bo zdarzenie ma wspolrzedne) — i przechodzi do nastepnego kroku.

### 5.3 Warunki zakonczenia rolloutu

Kazdy rollout moze zakonczyc sie na trzy sposoby:
- **Token GOAL** — posiadanie zakonczylo sie golem. Rollout jest deaktywowany i oznaczony jako sukces.
- **Token NO_GOAL** — posiadanie zakonczylo sie bez gola. Rollout jest deaktywowany i oznaczony jako porazka.
- **Osiagniecie max_seq_len** — sekwencja osiagnela maksymalna dlugosc. Rollout jest zakonczony bez wyniku (traktowany jako brak gola).

Deaktywowane rollouty nie generuja dalszych kroków, ale pozostaja w batchu do konca — ich wynik jest juz ustalony. Jesli wszystkie N rolloutow zakonczy sie wczesniej, petla przerywa sie bez koniecznosci generowania max_steps kroków. Dodatkowy parametr max_steps (domyslnie 10) ogranicza liczbe iteracji niezaleznie od max_seq_len.

Wszystkie N rolloutow jest przetwarzanych rownolegle jako batch na GPU — to nie jest petla po rolloutach, lecz zrownolegloowana operacja tensorowa. Przy N=1000 rolloutow kazdy krok wymaga jednego przebiegu (forward pass) przez Transformer dla batchu o rozmiarze 1000.

### 5.4 Dlaczego 3 zdarzenia na start?

Bierzemy pierwsze 3 zdarzenia z sekwencji jako kontekst startowy. Dlaczego 3? To kompromis:
- Za malo (1 zdarzenie) — model ma zbyt malo kontekstu, predykcje sa szumowe
- Za duzo (np. 8 zdarzen) — model juz "widzi" wiekszosc sekwencji, mniej musi generowac, wartość predykcyjna spada na sztucznie wywindowane poziomy

3 zdarzenia daja modelowi poczatkowy kontekst (skad zaczelo sie posiadanie, jaki byl pierwszy ruch), ale reszta sekwencji jest generowana — to wlasnie testuje zdolnosc modelu do "wyobrazania sobie" rozwoju akcji.

Jesli sekwencja ma 2 lub mniej zdarzen (wliczajac input), bierzemy tyle ile jest.

### 5.5 Liczba rollouts

Poczatkowo uzywalismy 100 rolloutów. Przy srednim xT okolo 0.1 (10% szans na gola) i 100 rolloutach, roznica jednego gola wiecej lub mniej zmienia wynik o 1 punkt procentowy — to istotny szum przy malych efektach. Dla finalnego porownania kluczowych wariantow (dual MDN, single MDN, pass-only) zwiekszylismy do 1000 rolloutów. Blad standardowy Monte Carlo spada proporcjonalnie do 1/sqrt(N) — z 1000 rolloutow szum jest ~3x mniejszy niz przy 100.

### 5.6 Ewaluacja po sekwencji, nie w batchu

Ewaluacja xT odbywa sie sekwencja po sekwencji (nie w batchu), poniewaz kazda sekwencja ma inna dlugosc kontekstu startowego i generuje rozna liczbe kroków. Dla kazdej z ~34 000 sekwencji walidacyjnych uruchamiamy osobna symulacje z N rolloutami. Przy N=1000 to daje okolo 34 miliony przebiegow modelu, co wymaga okolo 3 godzin obliczen na GPU (NVIDIA RTX) dla modeli carry i okolo 2 godzin dla modelu pass-only (krotsze sekwencje, mniejszy eval_max_seq_len).

### 5.7 Metryki ewaluacyjne

Na podstawie obliczonych wartosci xT dla kazdego posiadania oraz rzeczywistych etykiet (czy bylo gol czy nie) obliczamy:

- **ROC-AUC** — pole pod krzywa ROC. Mierzy zdolnosc modelu do rozrozniania posiadani zakonczonych golem od reszty. 0.5 = losowy, 1.0 = idealny.
- **Brier Score** — sredni kwadrat bledu miedzy xT a etykieta binarna. Mierzy kalibracke — czy model mowiacy "10% szans na gola" faktycznie ma racje w 10% przypadkow. Nizsza wartosc = lepiej.
- **Mean xT (goals)** — srednia wartosc xT dla posiadani, ktore zakonczyly sie golem. Wysoka wartosc = model poprawnie przypisuje wysokie zagrozenie przed golami.
- **Mean xT (no goals)** — srednia xT dla posiadani bez gola. Niska wartosc = model nie zawyza zagrozenia tam, gdzie gola nie bylo.
- **Separation** — roznica miedzy mean xT (goals) a mean xT (no goals). Im wieksza, tym lepiej model rozroznia zagrozenie. To kluczowa metryka z perspektywy uzytecznosci modelu w analizie pilkarskiej.
