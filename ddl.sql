create table quotes(
	close numeric,
	high numeric,
	interval int,
	low numeric,
	nb_of_trades int,
	open numeric,
	timestamp int,
	symbol varchar,
	volume numeric,
	primary key(symbol, timestamp, interval)
);
