Bitcoin Bias Predictor

This program is an automated tool that looks at the Bitcoin market and tries to predict where the price is heading next. It works by grabbing real-time price data for Bitcoin and analyzing it from a few different angles—looking at the big picture on the daily charts, down to the smaller hour-by-hour movements. 

To figure out if the market is looking bullish or bearish, it does a lot of heavy lifting behind the scenes. It manually calculates a few popular trading indicators, like the MACD and ATR, to gauge market momentum and volatility. On top of that, it scans the charts for institutional order blocks, which are basically strong zones where big players might be buying or selling.

After crunching all these numbers, it acts like a judge. It weighs the different signals against each other and comes up with a final prediction: Bullish, Bearish, or Neutral. Finally, it takes all this information and generates a clean, dark-themed visual report that automatically opens in your web browser so you can easily review the results.


Getting Started:

run.bat
This is the easiest way to start the program on Windows. Just double-click it. It will automatically set up everything it needs, download the required background tools, and run the main script for you.

build_exe.bat
If you want to turn the script into a standalone Windows app (an .exe file) that you can share or run without needing Python installed, double-click this. When it's done, you'll find your ready-to-run app inside a new folder called dist.
