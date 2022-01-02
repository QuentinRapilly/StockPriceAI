# pip install python-binance
from binance import Client
import config
import matplotlib.pyplot as plt
from IPython.display import display

class API():
    def __init__(self, Key, Security):
        self.client = Client(Key, Security)
        print("logged in")

    def livePrice(self, crypto="BTC", currency="USDT", timeStep=0.2, outputPrecision=0, maxToKeep=20):
        x = []
        y = []
        i = 0

        plt.style.use('dark_background')

        while True:
            # get latest price from Binance API
            coin = self.client.get_symbol_ticker(symbol=crypto+currency)
            print('\n',coin)

            plt.xlabel('Time since Start (s)')
            plt.ylabel('Price in '+currency)
            plt.title(crypto+' Real Time Price - Time Step = '+str(timeStep)+'s')

            x.append(i*timeStep)
            y.append(float(coin['price']))

            if len(x) > maxToKeep:
                x = x[-maxToKeep:]
                y = y[-maxToKeep:]

            plt.grid(alpha=0.3, linestyle='--')
            plt.plot(x, y, color='gold', linewidth=1, marker='o', markersize='5')


            for x_l,y_l in zip(x,y):

                if outputPrecision == 0:
                    label = int(y_l)
                else:
                    label = round(y_l, outputPrecision)

                plt.annotate(label, # this is the text
                            (x_l,y_l), # these are the coordinates to position the label
                            textcoords="offset points", # how to position the text
                            xytext=(0,10), # distance from text to points (x,y)
                            ha='center') # horizontal alignment can be left, right or center
        
            plt.pause(timeStep)
            plt.clf()
            i += 1
    

if __name__ == '__main__':
    apiKey = config.apiKey
    apiSecurity = config.apiSecurity

    api = API(apiKey, apiSecurity)
    api.livePrice()
