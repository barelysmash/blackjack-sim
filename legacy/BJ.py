import random
import numpy as np
from enum import Enum, auto

class Result(Enum): #RESULT OF EACH **HAND** NOT PLAYER (PLAYER[PL].HAND[H].END)
    NA = 0
    BLACKJACK = 1 #+1.5
    DDWIN = 2     #+2
    WIN = 3       #+1
    DDLOSE = 4    #-2
    LOSE = 5      #-1
    PUSH = 6      #0
    BUST = 7      #-1 #NOT IN USE 
    DDBUST = 8    #-2 #NOT IN USE
        
class DealerResult(Enum): #DEALER RESULT (DEALER.END)
    NA = 0
    BLACKJACK = 1
    STAND = 2
    BUST = 3
    
class Play(Enum):
    HIT = 1
    DOUBLEDOWN = 2  #REVERTS TO HIT IF CARDS > 2
    SPLIT = 3       #ONLY 2 CARDS, PAIRED
    STAND = 4
    DOUBLESTAND = 5 #ONLY A-7 against 3,4,5,6, REVERTS TO STAND IF CARDS > 2
    
basicStrategyMatrix = [ # NO ACE, NO PAIR, ANY NUMBER OF CARDS
    #DLR 2      3       4       5       6       7       8       9       10      A    handValue Row
    [Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1)],#5 R0
    [Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1)],#6 R1
    [Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1)],#7 R2
    [Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1),Play(1)],#8 R3
    [Play(1),Play(2),Play(2),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#9 R4
    [Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(1),Play(1)],#10 R5
    [Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(1),Play(1)],#11 R6
    [Play(1),Play(1),Play(4),Play(4),Play(4),Play(1),Play(1),Play(1),Play(1),Play(1)],#12 R7
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(1),Play(1),Play(1),Play(1),Play(1)],#13 R8
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(1),Play(1),Play(1),Play(1),Play(1)],#14 R9
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(1),Play(1),Play(1),Play(1),Play(1)],#15 R10
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(1),Play(1),Play(1),Play(1),Play(1)],#16 R11
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#17 R12
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#18 R13
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#19 R14
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#20 R15
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)] #21 R16
]

aceStrategyMatrix = [ # HAS ACE, NO PAIR, ANY NUMBER OF CARDS IF SUM(NON-ACES) <= 9***SOFT HAND***
    #   2       3       4       5       6       7       8       9       10      A
    [Play(1),Play(1),Play(1),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#A2(3,13)
    [Play(1),Play(1),Play(1),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#A3(4,14)
    [Play(1),Play(1),Play(2),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#A4(5,15)
    [Play(1),Play(1),Play(2),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#A5(6,16)
    [Play(1),Play(2),Play(2),Play(2),Play(2),Play(1),Play(1),Play(1),Play(1),Play(1)],#A6(7,17)
    [Play(4),Play(5),Play(5),Play(5),Play(5),Play(4),Play(4),Play(1),Play(1),Play(1)],#A7(8,18)
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#A8(9,19)
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#A9(10,20)
]

pairStrategyMatrix = [ # MATCHING PAIR, ONLY 2 CARDS
    #   2       3       4       5       6       7       8       9       10      A
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(1),Play(1),Play(1),Play(1)],#2s(4)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(1),Play(1),Play(1),Play(1)],#3s(6)
    [Play(1),Play(1),Play(1),Play(3),Play(3),Play(1),Play(1),Play(1),Play(1),Play(1)],#4s(8)
    [Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(2),Play(1),Play(1)],#5s(10)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(1),Play(1),Play(1),Play(1),Play(1)],#6s(12)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(1),Play(1),Play(1),Play(1)],#7s(14)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(1),Play(1)],#8s(16)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(4),Play(1),Play(1),Play(4),Play(4)],#9s(18)
    [Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4),Play(4)],#10s(20)
    [Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(3),Play(1)],#As(2,12)    
]

class Card():
    def __init__(self, face, value, suit, effect):
        self.face = face #face 2-10,J,Q,K,A
        self.value = value #value 1-11
        self.suit = suit #Heart,Club,Diamond,Spade
        self.effect = effect #-1,0,1

    def ShowCard(self):
        print(str(self.face) + " of " + str(self.suit))

class Deck():
    def __init__(self):
        self.cards = []
        self.InitDeck()

    def InitDeck(self):
        for suit in ['Hearts','Clubs','Diamonds','Spades']: #4 suits
            for face in [2,3,4,5,6,7,8,9,10,'J','Q','K','A']: #cards of each suit
                if face == 'A': #assign values for each card
                    value = 11
                elif face == 'J' or face == 'Q' or face == 'K':
                    value = 10
                else:
                    value = face
                if (value >= 2) and (value <= 6): #assign true count effect from card's value
                    effect = 1
                elif value >= 10:
                    effect = -1
                else:
                    effect = 0
                self.cards.append(Card(face,value,suit,effect)) #init Card with attributes and add to deck

class Hand():
    def __init__(self):
        self.cards = []      #cards in hand
        self.value = 0       #value of hand
        self.bet = 0         #bet on hand
        self.aces = False    #whether hand contains an ace
        self.pair = False    #whether 1st 2 cards match
        self.play = []       #sequence of play decisions
        self.nonAceVal = 0   #value of hand not including aces
        self.payout = 0      #payout of hand assigned after play ends
        self.result = 0      #result of play as a factor of bet (lose-1,push0,win+1,BJ+1.5,DDWIN+2,DDLOSE-2)
        self.BJ = False      #value == 21 with 1st 2 cards
        self.bust = False    #value > 21
        self.end = Result(0) #hand detail result

    def HandValue(self):
        self.aces = False
        self.pair = False
        aces = 0
        self.nonAceVal = 0
        values = []
        if (self.cards[0].face == self.cards[1].face and len(self.cards) == 2):
            self.pair = True # PAIRED HAND; USE FOR BASIC STRATEGY
        for card in self.cards:
            if card.face != 'A': 
                self.nonAceVal += card.value
            else: 
                aces += 1
                self.aces = True #SOFT HAND if aces > 0; USE FOR BASIC STRATEGY
        aceValues = Aces(aces)
        for i in aceValues:
            if i + self.nonAceVal <= 21:
                values.append(i + self.nonAceVal)
        if values == []:
            self.value = min(aceValues) + self.nonAceVal
        else:
            self.value = max(values)   

class Player():
    def __init__(self,cash):
        self.pHands = []           #array of hands to allow multiple hands if hands are split
        self.pHands.append(Hand()) #add to list of hands
        self.bankroll = cash       #starting table cash
        self.wager = 0             #temp value to store hand[0].bet after placing bets before resetting hands
        self.sumResult = []        #array of sums of all player's hand's results per round
        self.active = True         #active in current round of play
        self.seated = True         #bankroll exceeds minBet (use IF* player waits for target positive true count)

    def ResetHands(self):
        wager = self.pHands[0].bet #temp; b/c we reset in deal, after wagers made, deal only active players
        self.pHands = []           #empty any/all hands
        self.pHands.append(Hand()) #add hand to list of hands
        self.pHands[0].bet = wager #reassign bet to 1st hand after reinitializing hand
        
class Dealer():
    def __init__(self):
        self.hand = Hand()
        self.end = DealerResult(0)
        
    def ResetDealer(self):
        self.__init__()
               
def GenDeck():
    deck = Deck()
    random.shuffle(deck.cards)
    return deck

def GenShoe(noDecks):
    shoe = []
    for decks in range (0,noDecks): #6-deck shoe
        eachDeck = GenDeck()
        for card in range (0,len(eachDeck.cards)):
            eachCard = eachDeck.cards.pop()
            shoe.append(eachCard)
    random.shuffle(shoe)
    return shoe;

def CalcTrueCount(dealer, players, trueCount): #evaluate true count from visible cards each hand
    allCardEffects = 0
    for i in range(len(dealer.hand.cards)):
        allCardEffects += dealer.hand.cards[i].effect
    for pl in range(len(players)): #EACH PLAYER
        for h in range(len(players[pl].pHands)): #EACH HAND OF PLAYER
            for c in range(len(players[pl].pHands[h].cards)): #EACH CARD OF HAND
                allCardEffects += players[pl].pHands[h].cards[c].effect
    trueCount += allCardEffects
    return trueCount

#JUST MAKING SURE THAT THE OBSERVED TRUE COUNT AND SHOE TRUE COUNT ARE SAME. THEY ARE :)
#def ValidateTrueCount(shoe): #evaluate true count from cards remaining in shoe
#    trueCount = 0
#    for c in range (0,len(shoe)):
#        trueCount -= shoe[c].effect
#    return trueCount

def Aces(count): #build list of 2-option arrays for each ace in hand, then eval all permutations, return sums of ace values < 21
    opt = []
    for i in range(count):
        opt.append([1,11])
    perms = np.zeros((2 ** len(opt),len(opt))) #build ~truth table 2^#opts rows, #opts cols
    for j in range(len(opt)): #same code as autogenerating truth table
        n = len(opt) - j
        fill = int(2 ** (n - 1)) 
        for col in range(int(perms.shape[0] / fill / 2)):
            perms[col * 2 ** n : col * 2 ** n + fill, j] = 1
            perms[col * 2 ** n + fill : col * 2 ** n + fill * 2, j] = 11
        #print(perms) #show build of aces value permutation array
    return list(set([int(k) for k in np.sum(perms, axis = 1) if k <= 21])) #return list of unordered sets
    
def Deal(shoe,dealer,players):
    dealer.ResetDealer() #all hands reset with each deal
    for pl in range(len(players)):
        players[pl].ResetHands()
    for pl in range(len(players)):
        if players[pl].pHands[0].bet > 0:
            players[pl].active = True
    for j in range(2): #2 cards each OUTER LOOP
        for pl in range(len(players)): #INNER LOOP PLAYERS
            if players[pl].active == True:
                players[pl].pHands[0].cards.append(shoe.pop(0))
        dealer.hand.cards.append(shoe.pop(0)) #dealer's 1st card -- CASINO CARD, 2nd card hidden
    return shoe, dealer, players

def PlaceBets(minBet,bets,players,trueCount): #add parm 'results' if win/loss streak affects betting
    #bankRec = []
    for pl in range(len(players)):
        if players[pl].bankroll < minBet:
            players[pl].active = False
        else:
            if trueCount > 5:
                if players[pl].active == True:
                    players[pl].pHands[0].bet = 10 * round(((trueCount - 1) * players[pl].bankroll / 100) / 10) #bet in $10 increments
                    if players[pl].pHands[0].bet < minBet:
                        players[pl].pHands[0].bet = minBet
            else: # if true count is <= 1, play the table minimum
                if players[pl].active == True:
                    players[pl].pHands[0].bet = minBet
            players[pl].bankroll -= players[pl].pHands[0].bet
            bets[pl] = players[pl].pHands[0].bet #RECORD BETS EVEN FOR INACTIVE PLAYERS (FOR RESULTS ARRAY TABLE)
    print('Player bets:', bets)
    return bets, players

def BasicStrategy(shoe,dealerShows,player,results): #SINGLE PLAYER -- NOT ARRAY OF PLAYERS
    #MATRIX STRATEGY: PAIRS >> ACE >> BASIC
    blackjack = set([11,10])   #unordered set BJ to compare 2-card hands
    hands = len(player.pHands) # needed so that we can add to hands if play.SPLIT
    h = 0
    while h < hands:
        #if h < hands: #replaced for-in-range loop so strategy continues for added hand from split
        #print('hand/bet',h,player.pHands[h].bet)
        while player.pHands[h].value <= 21:
            if (set([player.pHands[h].cards[0].value,player.pHands[h].cards[1].value]) == blackjack and len(player.pHands[h].cards) == 2):
                player.pHands[h].BJ = True
                print('****** Player Blackjack ******')
            elif player.pHands[h].value < 21:
                #STRATEGY SELECTION
                if (player.pHands[h].pair == True and len(player.pHands[h].cards) == 2):
                    #print('pair',player.pHands[h].cards[0].value-2,dealerShows-2,'val',player.pHands[h].value)
                    player.pHands[h].play.append(pairStrategyMatrix[player.pHands[h].cards[0].value-2][dealerShows-2])
                    print(pairStrategyMatrix[player.pHands[h].cards[0].value-2][dealerShows-2])
                #elif (player.pHands[h].aces == True and len(player.pHands[h].cards) == 2):
                elif (player.pHands[h].aces == True and player.pHands[h].nonAceVal <= 9):
                    #print('ace',player.pHands[h].nonAceVal-2,dealerShows-2,'val',player.pHands[h].value)
                    player.pHands[h].play.append(aceStrategyMatrix[player.pHands[h].nonAceVal-2][dealerShows-2])
                    print(aceStrategyMatrix[player.pHands[h].nonAceVal-2][dealerShows-2])
                else:
                    #print('basic',player.pHands[h].value-5,dealerShows-2,'val',player.pHands[h].value)
                    player.pHands[h].play.append(basicStrategyMatrix[player.pHands[h].value-5][dealerShows-2])
                    print(basicStrategyMatrix[player.pHands[h].value-5][dealerShows-2])
                #VALIDATE SPECIAL CASES WITH ADDITIONAL BETS
                if player.bankroll < player.pHands[0].bet: #NO $$$, CANNOT DOUBLEDOWN, DOUBLESTAND, OR SPLIT:
                    if player.pHands[h].play[-1] == Play.DOUBLEDOWN:
                        player.pHands[h].play[-1] = Play.HIT
                    elif player.pHands[h].play[-1] == Play.DOUBLESTAND:
                        player.pHands[h].play[-1] = Play.STAND
                    elif player.pHands[h].play[-1] == Play.SPLIT:
                        player.pHands[h].play[-1] = basicStrategyMatrix[player.pHands[h].value-5][dealerShows-2]
                #VALIDATE/ADJUST STRATEGY
                if len(player.pHands[h].cards) > 2:
                    if player.pHands[h].play[-1] == Play.DOUBLEDOWN:
                        player.pHands[h].play[-1] = Play.HIT
                    elif player.pHands[h].play[-1] == Play.DOUBLESTAND:
                        player.pHands[h].play[-1] = Play.STAND
                #PERFORM VALIDATED STRATEGY
                if player.pHands[h].play[-1] == Play.HIT:
                    player.pHands[h].cards.append(shoe.pop(0))
                    player.pHands[h].cards[-1].ShowCard()
                elif player.pHands[h].play[-1] == Play.DOUBLEDOWN:
                    player.bankroll -= player.pHands[0].bet
                    player.pHands[h].bet += player.pHands[0].bet
                    player.pHands[h].cards.append(shoe.pop(0))
                    player.pHands[h].cards[-1].ShowCard()
                elif player.pHands[h].play[-1] == Play.DOUBLESTAND:
                    #print('*********************** DOUBLESTAND ***********************')
                    player.bankroll -= player.pHands[0].bet
                    player.pHands[h].bet += player.pHands[0].bet
                    player.pHands[h].cards.append(shoe.pop(0))
                    player.pHands[h].cards[-1].ShowCard()
                elif player.pHands[h].play[-1] == Play.SPLIT:
                    #print('******************** SPLIT ********************')
                    player.pHands.append(Hand())
                    hands += 1
                    #print('h, hands: ',h,hands)
                    player.bankroll -= player.pHands[0].bet
                    player.pHands[-1].bet = player.pHands[0].bet
                    splitCard = player.pHands[h].cards.pop(1)
                    player.pHands[h].cards.append(shoe.pop(0))
                    player.pHands[h].cards[0].ShowCard()
                    player.pHands[h].cards[1].ShowCard()
                    player.pHands[h].HandValue()
                    print(player.pHands[h].value)
                    player.pHands[-1].cards.append(splitCard)
                    player.pHands[-1].cards.append(shoe.pop(0))
                    player.pHands[-1].cards[0].ShowCard()
                    player.pHands[-1].cards[1].ShowCard()
                    player.pHands[-1].HandValue()
                    print(player.pHands[-1].value)
            player.pHands[h].HandValue()
            if (player.pHands[h].value >= 21 or \
                player.pHands[h].play[-1] == Play.STAND or \
                player.pHands[h].play[-1] == Play.DOUBLEDOWN or \
                player.pHands[h].play[-1] == Play.DOUBLESTAND):
                #print('** END HAND ** for loop',player.pHands[h].value)
                h += 1
                break #ONLY WAY TO EXIT LOOP
    for h in range(len(player.pHands)):
        print('** END HAND [' + str(h) + '] ** Final: ' + str(player.pHands[h].value))
        if player.pHands[h].value > 21:
            player.pHands[h].bust = True
    print('*** END PLAYER ***')
    return player, shoe

def Payout(dealer,players): #PAYOUT AND CALL RECORD FUNCTION TO ADD TO AGGREGATE RESULTS ARRAY
    if dealer.hand.value > 21:
        dealer.end = DealerResult(3) #dealer bust
    elif dealer.hand.BJ == True: 
        dealer.end = DealerResult(1) #dealer BJ
    else:
        dealer.end = DealerResult(2) #dealer stand
    for pl in range(len(players)):
        results = 0
        if players[pl].active == True: # only evaluate active players
            if dealer.hand.BJ == True: # dealer BJ
                if players[pl].pHands[0].BJ == True: #only need to evaluate player's first hand
                    players[pl].pHands[0].result = Result(6) #push against dealer BJ (no change to bet/payout/result)
                else:
                    players[pl].pHands[0].result = Result(5) #lose to dealer BJ
                    players[pl].pHands[0].bet = 0
                    results -= 1
            else: #no dealer BJ
                for h in range(len(players[pl].pHands)):
                    #print('>>>pay',pl,h,players[pl].pHands[h].bet) # CHECK INCOMING VALUES BEFORE PAYOUT EVAL
                    if players[pl].pHands[h].bust == True:
                        players[pl].pHands[h].bet = 0 #LOSE BET BUST LOSE/DOUBLELOSE
                        if (players[pl].pHands[h].play[-1] == Play.DOUBLEDOWN or players[pl].pHands[h].play[-1] == Play.DOUBLESTAND):
                            players[pl].pHands[h].result = Result(2) #doublelose bust
                            results -= 2
                        else:
                            players[pl].pHands[h].result = Result(5) #lose bust
                            results -= 1
                        #print('bust payout Hand Bet Pay',h,players[pl].pHands[h].bet,players[pl].pHands[h].payout)
                    elif players[pl].pHands[h].BJ == True:
                        players[pl].pHands[h].result = Result(1) #blackjack
                        players[pl].pHands[h].payout = (1.5 * players[pl].pHands[h].bet) #WIN PAYOUT 3:2
                        results += 1.5
                        #print('win BJ payout Hand Bet Pay',h,players[pl].pHands[h].bet,players[pl].pHands[h].payout)
                    elif ((players[pl].pHands[h].value > dealer.hand.value and players[pl].pHands[h].value <= 21) or \
                        (dealer.hand.value > 21 and players[pl].pHands[h].value <= 21)):
                        players[pl].pHands[h].payout = players[pl].pHands[h].bet # WIN PAYOUT 1:1 (bet includes DD)
                        if (players[pl].pHands[h].play[-1] == Play.DOUBLEDOWN or players[pl].pHands[h].play[-1] == Play.DOUBLESTAND):
                            players[pl].pHands[h].result = Result(2) #doublewin
                            results += 2
                        else:
                            players[pl].pHands[h].result = Result(3) #win
                            results += 1
                        #print('win std payout Hand Bet Pay',h,players[pl].pHands[h].bet,players[pl].pHands[h].payout)
                    elif (players[pl].pHands[h].value == dealer.hand.value and players[pl].pHands[h].value <= 21):
                        players[pl].pHands[h].result = Result(6) #push (no change to bet/result)
                    #elif (players[pl].pHands[h].value < dealer.hand.value and players[pl].pHands[h].value < 21):
                    else:
                        players[pl].pHands[h].bet = 0 #LOSE BET
                        if (players[pl].pHands[h].play[-1] == Play.DOUBLEDOWN or players[pl].pHands[h].play[-1] == Play.DOUBLESTAND):
                            players[pl].pHands[h].result = Result(4) #doublelose
                            results -= 2
                        else:
                            players[pl].pHands[h].result = Result(5) #lose
                            results -= 1
                        #print('lose std payout Hand Bet Pay',h,players[pl].pHands[h].bet,players[pl].pHands[h].payout)
                    print('Player/Hand/Result/Bet/Payout',pl,h,players[pl].pHands[h].result,players[pl].pHands[h].bet,players[pl].pHands[h].payout)
                    players[pl].bankroll += (players[pl].pHands[h].bet + players[pl].pHands[h].payout) #return pushes and payouts
                    players[pl].pHands[h].bet = 0
        players[pl].sumResult.append(results)
        print('Player ' + str(pl) + ' bank: ' + str(players[pl].bankroll) + ' results: ' + str(results))
    return players

def main():
    #### ADJUSTABLE VARIABLES ###
    shoes = 3                #no. shoes to simulate
    noDecks = 6              #no. decks in shoe (default 6)
    noPlayers = 3            #no. players
    minBet = 5               #minimum wager
    cash = 500               #initial funds for all players
    reshuffle = (noDecks * 52) * 0.25 #reshuffle when 25% of shoe remains
    #trueCountThreshhold = 5  #only adjust bet if true count exceeds this value
    #betIncrement = 10        #increments of bet rounded to this if exceeding true count threshhold
    #betParts = 100           #increments of bet are function of 1/betParts (100 = increment is 1% of bankroll)
    ### NO CHANGES BELOW THIS LINE ###
    #betStyle = [trueCountThreshhold,betIncrement,betParts]
    bets = []
    bets = [0 for j in range(noPlayers)] #bets array ONLY USED TO SHOW ALL BETS WITHOUT ITERATING PLAYERS
    results = []
    results = [0 for k in range(noPlayers)] #single round results array
    players = []
    dealer = Dealer()
    for pl in range(noPlayers):
        addPlayer = Player(cash)
        players.append(addPlayer)
    blackjack = set([11,10]) #unordered set BJ to compare 2-card hands
    for each in range(shoes):
        trueCount = 0
        shoe = GenShoe(noDecks)
        while len(shoe) > (reshuffle):
            print('************ HAND ************')
            bets, players = PlaceBets(minBet,bets,players,trueCount) #bet is attribute of each player's hand
            shoe, dealer, players = Deal(shoe,dealer,players) #ALL HANDS ARE RESET IN DEAL()
            print('Dealer shows:')
            dealer.hand.cards[0].ShowCard() #casino card
            #dealer.hand.cards[1].ShowCard() #down card # REMOVE AFTER TEST
            dealer.hand.HandValue()
            #print(dealer.hand.value)                   # REMOVE AFTER TEST
            for pl in range(noPlayers): # SHOW PLAYER CARDS
                if players[pl].active:
                    print('Player[' + str(pl) + ']')
                    players[pl].pHands[0].cards[0].ShowCard()
                    players[pl].pHands[0].cards[1].ShowCard()
                    players[pl].pHands[0].HandValue() #CALC VALUE TO INITIAL HAND
                    print(players[pl].pHands[0].value)
            if set([dealer.hand.cards[0].value,dealer.hand.cards[1].value]) == blackjack: #CHECK FOR DEALER BJ
                dealer.hand.BJ = True
                print('****** Dealer Blackjack ******')
                for pl in range(noPlayers):
                    if players[pl].active:
                        if set([players[pl].pHands[0].cards[0].value,players[pl].pHands[0].cards[1].value]) != blackjack: #PLAYER LOSES IF NO BJ
                            results[pl] += -1
                        else:
                            players[pl].pHands[0].BJ = True
                            print('*** PUSH * Player[' + str(pl) + '] Blackjack ***') #PLAYER PUSHES IF ALSO BJ
            else: # IF NO DEALER BJ...
                dealerPlay = 0
                for pl in range(noPlayers):
                    if players[pl].active:
                        print('Player[' + str(pl) + '] bet: ' + str(players[pl].pHands[0].bet) + ' hand: ' \
                             + str(players[pl].pHands[0].value))
                        players[pl], shoe = BasicStrategy(shoe,dealer.hand.cards[0].value,players[pl],results)
                        for h in range(len(players[pl].pHands)):
                            if players[pl].pHands[h].bust == False: # CHECK FOR LIVE HANDS
                                dealerPlay += 1
                if dealerPlay > 0: #IF NO LIVE HANDS, NO DEALER PLAY 
                    print('Dealer Showdown:', dealer.hand.value)
                    while dealer.hand.value < 17: #DEALER PLAY
                        print('Dealer.HIT on',dealer.hand.value)
                        dealer.hand.cards.append(shoe.pop(0))
                        dealer.hand.HandValue()
                        dealer.hand.cards[-1].ShowCard() #show most recent card drawn
                    if dealer.hand.value > 21:
                        dealer.hand.bust = True
                    print('Dealer Final: ' + str(dealer.hand.value))
                else:
                    print('****** All Players Bust ******')
            players = Payout(dealer,players)
            trueCount = CalcTrueCount(dealer, players, trueCount) #adjust true count after play ends
            print('    *** True Count: ' + str(trueCount) + ' ***')
    for pl in range(noPlayers):
        totalResult = 0
        totalRounds = 0
        for r in range(len(players[pl].sumResult)):
            totalResult += players[pl].sumResult[r]
            totalRounds += 1
        print('Player[' + str(pl) + '] Results: ' + str(totalResult) + ' Bank: $' + str(players[pl].bankroll)\
              + ' Return: ' + str(round(((players[pl].bankroll/cash - 1) * 100),2)) + '%')
        #print('Player Results Bank',pl,totalResult,players[pl].bankroll)
    print('Rounds:',totalRounds)
    
main()  
