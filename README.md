# Riding the Trend Momentum Trading Bot 
Momentum based trading bot that rides the trend using Exponential Moving Average Crossover.
This bot is built with [Python](https://www.python.org/) as the native language has and [MT5](https://www.metatrader5.com/en) Integrations for real time streaming of market data and trade executions.

**Core functionalities of the Bot:**

- Connects to any MT5 broker — demo or live
- Scans for EMA crossover signals on a configurable interval
- Automatically opens, manages, and closes trades
- Sizes each position dynamically based on your account balance and risk tolerance
- Protects every trade with Stop Loss, Take Profit, and optional Trailing Stop
- Halts automatically if your maximum drawdown threshold is breached
- Sends real-time Telegram alerts for every trade event
- Provides a full Telegram control panel to start, stop, and monitor the bot from your phone
- Runs vectorised backtests and parameter optimisation on demand

**How the EMA crossover signal works:**

```
EMA Short (fast) crosses ABOVE EMA Long (slow)  →  BUY   🟢
EMA Short (fast) crosses BELOW EMA Long (slow)  →  SELL  🔴
No crossover detected on last closed candle      →  HOLD  ⚪
```

The signal is always read from the **last closed candle** — never the live open candle — to prevent repainting and false entries.



## Forward Testing and Trade Results
Started forward testing the trading bot with $5000 and the account has grown to $6483.35, that's 129.67% profit from the initial capital 

![01](./images/TradStats1.png)
![02](./images/TradStats2.png)
![03](./images/TradStats3.png)
![04](./images/TradStats4.png)
![05](./images/TradStats5.png)
![06](./images/TradStats6.png)
![07](./images/TradStats7.png)
![08](./images/TradStats8.png)
![09](./images/TradStats9.png)
![10](./images/Tradelog.jpg)




---