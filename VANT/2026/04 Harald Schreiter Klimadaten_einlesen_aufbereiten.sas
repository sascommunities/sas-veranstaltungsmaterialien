/*******************************************************************************************************/
/*******************************************************************************************************/
/********      Einlesen von Jahresmittel-Temperaturen für die Bundesländer in Deutschland        *******/
/*******************************************************************************************************/
/*******************************************************************************************************/
filename dwd_ytmp temp encoding="utf-8";

proc http
	url="https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/air_temperature_mean/regional_averages_tm_year.txt"
	method="GET"
	out=dwd_ytmp;
run;

/*******************************************************************************************************/
/* Daten in SAS Data Set WORK.TEMP_GER_HISTORY importieren                                             */
/*******************************************************************************************************/
proc import file=dwd_ytmp out=work.temp_ger_history replace dbms=dlm;
            datarow=3;  
            delimiter=";";
            getnames=no;
run;

/*******************************************************************************************************/
/* Daten zur weiteren Verwendung aufbereiten                                                           */
/*******************************************************************************************************/
proc sql;
	create table work.ger_temperaturen_jahr
		as select 
			mdy(01, 01, var1) as Jahr format year4.,
			var3 as Berlin,
			var4 as Brandenburg,
			var5 as Baden_Wuerttemberg,
			var6 as Bayern,
			var7 as Hessen,
			var8 as Mecklenburg_Vorpommern,
			var9 as Niedersachsen,
			var10 as Hamburg_Bremen,
			var11 as Nordrhein_Westfalen,
			var12 as Rheinland_Pfalz,
			var13 as Schleswig_Holstein,
			var14 as Saarland,
			var15 as Sachsen,
			var16 as Sachsen_Anhalt,
			var18 as Thueringen,
			var19 as Deutschland
		from work.temp_ger_history;
quit;

/*******************************************************************************************************/
/* notwendige Sortierung vor PROC TRANSPOSE                                                            */
/*******************************************************************************************************/
proc sort data=work.ger_temperaturen_jahr out=work.ger_temp_yearly;
	by Jahr;
run;

/*******************************************************************************************************/
/* Daten Transponieren für eine Gruppenspalte REGION und eine TEMPERATUR Spalte                        */
/*******************************************************************************************************/
proc transpose data=work.ger_temp_yearly out=work.ger_temp_yearly_T1 
				(rename=(_name_=Region COL1=AVG_Temp_Year));
	by Jahr;
	var Baden_Wuerttemberg Bayern Berlin Brandenburg Deutschland Hamburg_Bremen
		Hessen Niedersachsen Nordrhein_Westfalen Rheinland_Pfalz Saarland
		Sachsen Sachsen_Anhalt Schleswig_Holstein Thueringen Mecklenburg_Vorpommern;
run;

/*******************************************************************************************************/
/* Temperatur Spalte einheitlich formatieren, Dekade berechnen                                         */
/*******************************************************************************************************/
data work.dwd_yearly_temp;
	format Dekade $6. Jahr_txt $4.;
	set work.ger_temp_yearly_T1;
	Dekade = (substr(put(Jahr, year4.), 1, 3) || "0er");
	Jahr_txt = put(Jahr, year4.);
	format AVG_Temp_year commax6.2;
run;



/*******************************************************************************************************/
/*******************************************************************************************************/
/************      Einlesen von Anzahl Frosttage für die Bundesländer in Deutschland       *************/
/*******************************************************************************************************/
/*******************************************************************************************************/
filename dwd_ycld temp encoding="utf-8";

proc http
	url="https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/frost_days/regional_averages_tnas_year.txt"
	method="GET"
	out=dwd_ycld;
run;

/*******************************************************************************************************/
/* Daten in SAS Data Set WORK.COLD_GER_HISTORY importieren                                             */
/*******************************************************************************************************/
proc import file=dwd_ycld out=work.cold_ger_history replace dbms=dlm;
            datarow=3;  
            delimiter=";";
            getnames=no;
run;

/*******************************************************************************************************/
/* Daten zur weiteren Verwendung aufbereiten                                                           */
/*******************************************************************************************************/
proc sql;
	create table work.ger_colddays_jahr
		as select 
			mdy(01, 01, var1) as Jahr format year4.,
			var3 as Berlin,
			var4 as Brandenburg,
			var5 as Baden_Wuerttemberg,
			var6 as Bayern,
			var7 as Hessen,
			var8 as Mecklenburg_Vorpommern,
			var9 as Niedersachsen,
			var10 as Hamburg_Bremen,
			var11 as Nordrhein_Westfalen,
			var12 as Rheinland_Pfalz,
			var13 as Schleswig_Holstein,
			var14 as Saarland,
			var15 as Sachsen,
			var16 as Sachsen_Anhalt,
			var18 as Thueringen,
			var19 as Deutschland
		from work.cold_ger_history;
quit;

/*******************************************************************************************************/
/* notwendige Sortierung vor PROC TRANSPOSE                                                            */
/*******************************************************************************************************/
proc sort data=work.ger_colddays_jahr out=work.ger_colddays_jahr;
	by Jahr;
run;

/*******************************************************************************************************/
/* Daten Transponieren für eine Gruppenspalte REGION und eine FROSTTAGE Spalte                        */
/*******************************************************************************************************/
proc transpose data=work.ger_colddays_jahr out=work.ger_colddays_jahr_T1 
				(rename=(_name_=Region COL1=Frosttage_Year));
	by Jahr;
	var Baden_Wuerttemberg Bayern Berlin Brandenburg Deutschland Hamburg_Bremen
		Hessen Niedersachsen Nordrhein_Westfalen Rheinland_Pfalz Saarland
		Sachsen Sachsen_Anhalt Schleswig_Holstein Thueringen Mecklenburg_Vorpommern;
run;

/*******************************************************************************************************/
/* Frosttage Spalte einheitlich formatieren, Dekade berechnen                                          */
/*******************************************************************************************************/
data work.dwd_yearly_cold;
	format Dekade $6. Jahr_txt $4.;
	set work.ger_colddays_jahr_T1;
	Dekade = (substr(put(Jahr, year4.), 1, 3) || "0er");
	Jahr_txt = put(Jahr, year4.);
	format Frosttage_Year commax6.1;
run;


/*******************************************************************************************************/
/*******************************************************************************************************/
/************   Einlesen von Anzahl Hitzetage min 30° für die Bundesländer in Deutschland   ************/
/*******************************************************************************************************/
/*******************************************************************************************************/
filename dwd_yhot temp encoding="utf-8";

proc http
	url="https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/hot_days/regional_averages_txbs_year.txt"
	method="GET"
	out=dwd_yhot;
run;

/*******************************************************************************************************/
/* Daten in SAS Data Set WORK.hot_GER_HISTORY importieren                                             */
/*******************************************************************************************************/
proc import file=dwd_yhot out=work.hot_ger_history replace dbms=dlm;
            datarow=3;  
            delimiter=";";
            getnames=no;
run;

/*******************************************************************************************************/
/* Daten zur weiteren Verwendung aufbereiten                                                           */
/*******************************************************************************************************/
proc sql;
	create table work.ger_hotdays_jahr
		as select 
			mdy(01, 01, var1) as Jahr format year4.,
			var3 as Berlin,
			var4 as Brandenburg,
			var5 as Baden_Wuerttemberg,
			var6 as Bayern,
			var7 as Hessen,
			var8 as Mecklenburg_Vorpommern,
			var9 as Niedersachsen,
			var10 as Hamburg_Bremen,
			var11 as Nordrhein_Westfalen,
			var12 as Rheinland_Pfalz,
			var13 as Schleswig_Holstein,
			var14 as Saarland,
			var15 as Sachsen,
			var16 as Sachsen_Anhalt,
			var18 as Thueringen,
			var19 as Deutschland
		from work.hot_ger_history;
quit;

/*******************************************************************************************************/
/* notwendige Sortierung vor PROC TRANSPOSE                                                            */
/*******************************************************************************************************/
proc sort data=work.ger_hotdays_jahr out=work.ger_hotdays_jahr;
	by Jahr;
run;

/*******************************************************************************************************/
/* Daten Transponieren für eine Gruppenspalte REGION und eine HitzeTAGE Spalte                        */
/*******************************************************************************************************/
proc transpose data=work.ger_hotdays_jahr out=work.ger_hotdays_jahr_T1 
				(rename=(_name_=Region COL1=Hitzetage_Year));
	by Jahr;
	var Baden_Wuerttemberg Bayern Berlin Brandenburg Deutschland Hamburg_Bremen
		Hessen Niedersachsen Nordrhein_Westfalen Rheinland_Pfalz Saarland
		Sachsen Sachsen_Anhalt Schleswig_Holstein Thueringen Mecklenburg_Vorpommern;
run;

/*******************************************************************************************************/
/* Hitzetage Spalte einheitlich formatieren, Dekade berechnen                                          */
/*******************************************************************************************************/
data work.dwd_yearly_hot;
	format Dekade $6. Jahr_txt $4.;
	set work.ger_hotdays_jahr_T1;
	Dekade = (substr(put(Jahr, year4.), 1, 3) || "0er");
	Jahr_txt = put(Jahr, year4.);
	format Hitzetage_Year commax6.1;
run;

/*******************************************************************************************************/
/*******************************************************************************************************/
/************      Einlesen von Anzahl Niederschlag für die Bundesländer in Deutschland    *************/
/*******************************************************************************************************/
/*******************************************************************************************************/
filename dwd_ypre temp encoding="utf-8";

proc http
	url="https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/precipitation/regional_averages_rr_year.txt"
	method="GET"
	out=dwd_ypre;
run;

/*******************************************************************************************************/
/* Daten in SAS Data Set WORK.prec_GER_HISTORY importieren                                             */
/*******************************************************************************************************/
proc import file=dwd_ypre out=work.prec_ger_history replace dbms=dlm;
            datarow=3;  
            delimiter=";";
            getnames=no;
run;

/*******************************************************************************************************/
/* Daten zur weiteren Verwendung aufbereiten                                                           */
/*******************************************************************************************************/
proc sql;
	create table work.ger_prec_jahr
		as select 
			mdy(01, 01, var1) as Jahr format year4.,
			var3 as Berlin,
			var4 as Brandenburg,
			var5 as Baden_Wuerttemberg,
			var6 as Bayern,
			var7 as Hessen,
			var8 as Mecklenburg_Vorpommern,
			var9 as Niedersachsen,
			var10 as Hamburg_Bremen,
			var11 as Nordrhein_Westfalen,
			var12 as Rheinland_Pfalz,
			var13 as Schleswig_Holstein,
			var14 as Saarland,
			var15 as Sachsen,
			var16 as Sachsen_Anhalt,
			var18 as Thueringen,
			var19 as Deutschland
		from work.prec_ger_history;
quit;

/*******************************************************************************************************/
/* notwendige Sortierung vor PROC TRANSPOSE                                                            */
/*******************************************************************************************************/
proc sort data=work.ger_prec_jahr out=work.ger_prec_jahr;
	by Jahr;
run;

/*******************************************************************************************************/
/* Daten Transponieren für eine Gruppenspalte REGION und eine HitzeTAGE Spalte                        */
/*******************************************************************************************************/
proc transpose data=work.ger_prec_jahr out=work.ger_prec_jahr_T1 
				(rename=(_name_=Region COL1=Niederschlag_Year));
	by Jahr;
	var Baden_Wuerttemberg Bayern Berlin Brandenburg Deutschland Hamburg_Bremen
		Hessen Niedersachsen Nordrhein_Westfalen Rheinland_Pfalz Saarland
		Sachsen Sachsen_Anhalt Schleswig_Holstein Thueringen Mecklenburg_Vorpommern;
run;

/*******************************************************************************************************/
/* Niederschlag Spalte einheitlich formatieren, Dekade berechnen                                          */
/*******************************************************************************************************/
data work.dwd_yearly_prec;
	format Dekade $6. Jahr_txt $4.;
	set work.ger_prec_jahr_T1;
	Dekade = (substr(put(Jahr, year4.), 1, 3) || "0er");
	Jahr_txt = put(Jahr, year4.);
	format Niederschlag_Year commax8.1;
run;



/*******************************************************************************************************/
/* jährliche Daten (TEMP COLD HOT PRECIP) zusammenführen                                                                      */
/*******************************************************************************************************/

proc SQL; 
create table dwd_yearly_all as
select temp.Dekade,
		temp.Jahr_txt,
		temp.Jahr,
		temp.Region,
		temp.AVG_Temp_Year,
		cold.Frosttage_Year,
		hot.Hitzetage_Year,
		prec.Niederschlag_Year
from work.dwd_yearly_temp as temp
left join work.dwd_yearly_cold as cold
on temp.jahr_txt = cold.jahr_txt and temp.region = cold.region
left join work.dwd_yearly_hot as hot
on temp.jahr_txt = hot.jahr_txt and temp.region = hot.region
left join work.dwd_yearly_prec as prec
on temp.jahr_txt = prec.jahr_txt and temp.region = prec.region
;
quit;


/*******************************************************************************************************/
/* Finales Aufräumen der Zwischentabellen aus den jährlichen Datenquellen                              */
/*******************************************************************************************************/
proc datasets lib=WORK nolist;
	delete ger_temperaturen_jahr ger_temp_yearly ger_temp_yearly_T1 temp_ger_history;
	delete ger_colddays_jahr cold_ger_history ger_colddays_jahr_t1 cold_ger_hiytory;
	delete ger_hotdays_jahr hot_ger_history ger_hotdays_jahr_t1 hot_ger_hiytory;
	delete ger_prec_jahr prec_ger_history ger_prec_jahr_t1 prec_ger_hiytory;
run;

filename dwd_ytmp clear;
filename dwd_ycld clear;
filename dwd_yhot clear;
filename dwd_ypre clear;




/*******************************************************************************************************/
/* 12 einzelne Monatstabellen einlesen und in eine Gesamttabelle                                       */
/* schreiben incl. Transponieren der Daten für späteres Zusammenführen mit den anderen Daten           */ 
/*******************************************************************************************************/
/* Datenquelle                                                                                         */
/* https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/  */
/* kopieren der 12(!) einzelnen TXT Files nach c:\temp\DWD                                             */ 
/*******************************************************************************************************/

/********************************************************************/
/* falls Zieltabelle schon besteht, diese löschen                   */
/********************************************************************/
proc datasets lib=work nolist nodetails;
	delete ger_temp_monthly;
run;

/********************************************************************/
/* Macro zum Einlesen der 12 einzelnen Monatstabellen               */
/********************************************************************/
%macro dwd_mon;

%do i=1 %to 12;
/* Import der 12 einzelnen Datenquellen */ 

%let i=%sysfunc(putn(&i,Z2.));

filename dwd_mon TEMP;

proc http
	url="https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/regional_averages_tm_&i..txt"
	method="GET"
	out=dwd_mon;
run;

proc import file=dwd_mon out=work.temp_ger_mon_&i replace dbms=dlm;
            datarow=3;  
            delimiter=";";
            getnames=no;
run;

/* Namen der Spalten korrekt zuordnen */ 
proc sql;
	create table work.ger_temp_&i
		as select 
			mdy(var2, 1, var1) as Datum format=date9.,
			var3 as Berlin,
			var4 as Brandenburg,
			var5 as Baden_Wuerttemberg,
			var6 as Bayern,
			var7 as Hessen,
			var8 as Mecklenburg_Vorpommern,
			var9 as Niedersachsen,
			var10 as Hamburg_Bremen,
			var11 as Nordrhein_Westfalen,
			var12 as Rheinland_Pfalz,
			var13 as Schleswig_Holstein,
			var14 as Saarland,
			var15 as Sachsen,
			var16 as Sachsen_Anhalt,
			var18 as Thueringen,
			var19 as Deutschland
		from work.temp_ger_mon_&i ;
quit;

/* neue Tabelle an (leere) Zieltabelle anhängen */ 
proc append base=work.ger_temp_monthly data=work.ger_temp_&i force nowarn;
run;

/* jeweils 12 einzelne temp. Data Sets aufräumen*/
proc datasets lib=work nolist nodetails;
	delete ger_temp_&i temp_ger_mon_&i;
run;

filename dwd_mon clear;

%end;

%mend dwd_mon;

/* Macro ausführen zur Erstellung der Gesamttabelle */
%dwd_mon;

/********************************************************************/
/* notwendiges Sortieren vor PROC TRANSPOSE                         */
/********************************************************************/
proc sort data=work.ger_temp_monthly;
	by Datum;
run;

/********************************************************************/
/* Daten Transponieren für Gruppenspalte REGION u. TEMPERATUR       */ 
/********************************************************************/
proc transpose data=work.ger_temp_monthly 
	out=work.ger_temp_monthly_T1(rename=(_name_=Region COL1=AVG_Temp_mon));
	by datum;
	var Baden_Wuerttemberg Bayern Berlin Brandenburg Deutschland Hamburg_Bremen
		Hessen Niedersachsen Nordrhein_Westfalen Rheinland_Pfalz Saarland
		Sachsen Sachsen_Anhalt Schleswig_Holstein Thueringen Mecklenburg_Vorpommern;
run;

/********************************************************************/
/* zusätzliche Spalten berechnen und formatieren                    */ 
/********************************************************************/
data work.dwd_monthly_temp;
	set work.ger_temp_monthly_T1;
	format Monat NLDATEMN. Jahr_txt $4. Monat_txt $20. AVG_Temp_mon commax6.2; 
	label Monat="Meß-Zeitraum AVG Temp." Monat_txt="Monats-Nummer";
	Monat = datum;
	Monat_txt = put(month(datum), z2.); 
	Jahr_txt  = put (year(datum), 4.);
run;

/********************************************************************/
/* Aufräumen der Zwischentabellen                                   */
/********************************************************************/
proc datasets lib=WORK nolist;
	delete ger_temp_monthly GER_TEMP_MONTHLY_T1;
run;

/*********************************************************************************************/
/* monatliche CO2 Konzentraion in PPM gemessen auf dem Mauna Loa, Hawaii (seit 1958)         */
/* Quelle: https://datahub.io/core/co2-ppm/_r/-/data/co2-mm-mlo.csv                          */
/* Datei in das Verzeichnis C:\TEMP\DWD kopieren                                             */
/*********************************************************************************************/
/* weitere Quelle: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv               */
/*********************************************************************************************/
/* Einlesen von monatliche CO2 Konzentrationen weltweit                                      */
/*********************************************************************************************/
filename co2_data temp encoding="utf-8";

proc http
	url="https://datahub.io/core/co2-ppm/_r/-/data/co2-mm-mlo.csv"
	method="GET"
	out=co2_data;
run;

/*********************************************************************************************/
/* Daten in SAS Data Set importieren                                                         */
/*********************************************************************************************/
proc import file=co2_data out=work.co2_mon replace dbms=dlm;
            datarow=2;  
            delimiter=",";
            getnames=yes;
run;

/*********************************************************************************************/
/* relevante Spalten selektieren und umbenennen                                              */ 
/*********************************************************************************************/
data work.co2_world_mon(keep=Date CO2_AVG_PPM);
	set work.co2_mon;
	rename Average=CO2_AVG_PPM;
run;

/********************************************************************/
/* Aufräumen der Zwischentabellen                                   */
/********************************************************************/
proc datasets lib=WORK nolist;
	delete co2_mon;
run;


/*********************************************************************************************/
/*********************************************************************************************/
/* einzelne Temperaturen (jährlich/monatlich) und CO2 Konzentrations-Tabellen joinen         */
/*********************************************************************************************/
/*********************************************************************************************/
PROC SQL;
   CREATE TABLE WORK.KLIMADATEN_MONATLICH_GESAMT AS 
   SELECT t1.Datum LABEL="Datum der Messung" AS Datum, 
          t1.Monat LABEL="Monat der Messung als Datum" AS Monat, 
          t1.Monat_txt LABEL="Monats-Nummer als Text" AS Monat_txt, 
          /* Jahr */
            (mdy (01, 01,year(datum))) FORMAT=year4. LABEL="Jahr der Messung als Datum" AS Jahr, 
          t1.Jahr_txt LABEL="Jahr als Text" AS Jahr_txt, 
          t2.Dekade LABEL="Dekade" AS Dekade, 
          t1.Region LABEL="Region" AS Region, 
          t1.AVG_Temp_mon LABEL="monatliche Durchschnittstemperatur" AS AVG_Temp_mon, 
          t2.AVG_Temp_year LABEL="jährliche Durchschnittstemperatur" AS AVG_Temp_year, 
          t3.CO2_AVG_PPM LABEL="monatliche CO2 Durchschnittskonzentration in PPM" AS CO2_AVG_PPM_mon
      FROM WORK.DWD_MONTHLY_TEMP t1
           LEFT JOIN WORK.DWD_YEARLY_TEMP t2 ON (t1.Jahr_txt = t2.Jahr_txt) AND (t1.Region = t2.Region)
           LEFT JOIN WORK.CO2_WORLD_MON t3 ON (t1.Datum = t3.Date)
      ORDER BY t1.Datum,
               t1.Region;
QUIT;

/**********************************************************************************************/
/* Monatliche Mitteltemperaturen für Deutschland auf Dekaden Mittel verdichten und bereinigen */ 
/**********************************************************************************************/
proc means data=work.KLIMADATEN_MONATLICH_GESAMT n mean nonobs noprint;
	class dekade monat_txt;
	var AVG_Temp_Mon;
	output out=work.KLIMADATEN_MON_GRP mean=Mon_Temp_Mean;
	where Region = "Deutschland";
run;

proc sql;
	create table work.KLIMADATEN_MON_GRP_DEKADE as
	select Dekade,
		Monat_txt,
		Mon_Temp_Mean label="Dekaden Monatsmittel-Temperatur"
	from work.KLIMADATEN_MON_GRP
		where _TYPE_ = 3;
quit;

proc datasets lib=WORK nolist;
	delete KLIMADATEN_MON_GRP;
run;



/************************************************************************************************/
/* Import WORLD Ocean: tägliche Meeresoberfläche Temperatur 60°S-60°N, 0-360°E                  */
/* Quelle: https://climatereanalyzer.org/clim/sst_daily/json_2clim/oisst2.1_world2_sst_day.json */
/************************************************************************************************/
/************************************************************************************************/
/* Einlesen von tägliche Temperaturen Meeresoberfläche                                          */
/************************************************************************************************/
filename world_oc temp encoding="utf-8";

proc http
	url="https://climatereanalyzer.org/clim/sst_daily/json_2clim/oisst2.1_world2_sst_day.json"
	method="GET"
	out=world_oc;
run;

/************************************************************************************************/
/* Daten in SAS Data Set importieren                                                            */
/************************************************************************************************/
libname ocean json fileref=world_oc;

/************************************************************************************************/
/* relevante Daten des JSON Files zusammenführen: Temperatur und Jahre                          */
/************************************************************************************************/
PROC SQL noprint;
   CREATE TABLE work.world_ocean_temp AS 
   SELECT t2.name, 
          t1.* 
      FROM OCEAN.DATA t1
           INNER JOIN OCEAN.ROOT t2 ON (t1.ordinal_root = t2.ordinal_root)
      WHERE t2.name > '1982';
QUIT; 

/************************************************************************************************/
/* notwendige Sortierung vor dem Transponieren                                                  */
/************************************************************************************************/
proc sort data=work.world_ocean_temp;
	by name;
run;

/************************************************************************************************/
/* Die 366 Tagesspalten in eine Temperatur Spalte transponieren                                 */ 
/************************************************************************************************/
proc transpose data=work.world_ocean_temp
	out=work.world_ocean_temp_2(rename=(_name_=TagDesJahres COL1=AVG_Temp_Day));
	by name;
	var data1-data366;
run;

/***********************************************************************************************/
/* Bereinigen der TagDesJahres Spalte um den String "data" sowie Format Anpassung              */
/***********************************************************************************************/
data work.world_ocean_temp_daily;
	set work.world_ocean_temp_2;
	format AVG_Temp_Day commax6.2;
	Label TagDesJahres = "Tag des Jahres";
	rename name=Zeitraum;
	TagDesJahres = compress(TagDesJahres, "data", "");
	   where substr(name,1,2) = "19" or substr(name,1,2) = "20";
run;

data work.world_ocean_temp_daily;
	set work.world_ocean_temp_daily;
	format Datum date9.;
		if Zeitraum = "1982-2010" or Zeitraum = "1991-2020" then Datum = .;
		else Datum = intnx('day', mdy(1,1,input(Zeitraum, 4.)), TagDesJahres);
		/* wenn Temp fehlt - wegen 366 Tage - mit dem vorherigen Wert füllen */
		if AVG_Temp_Day = . then AVG_Temp_Day=lag(AVG_Temp_Day);
run;

/***********************************************************************************************/
/* Aufräumen der Zwischentabellen                                                              */
/***********************************************************************************************/
proc datasets lib=WORK nolist;
	delete world_ocean_temp world_ocean_temp_2;
run;




