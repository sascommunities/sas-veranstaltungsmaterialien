/********************************************************************/
/********************************************************************/
/*************  Ausgabe der Daten in diversen Charts   **************/
/********************************************************************/
/********************************************************************/

ods graphics on / width=1400 height=800;

*ods pdf file="/home/u64257992/Klimacharts.pdf" style=seaside;
*options orientation=landscape papersize=A4;

/**************************************************************************/
/* Zeitreihen Diagramm für Monatsmittel CO2 PPM auf dem Mauna Loa, Hawaii */
/**************************************************************************/
footnote "Quelle: https://datahub.io/core/co2-ppm/_r/-/data/co2-mm-mlo.csv";

proc sgplot data=work.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Monatliche Durchschnittswerte der CO2 PPM auf dem Mauna Loa, Hawaii";
	title2 height=08pt "Daten liegen erst ab März 1958 vor. Saisonale Schwankungen durch unstetes saisonales Pflanzenwachstum.";
	series x=Datum y=CO2_AVG_PPM_mon / markers
		markerfillattrs=(COLOR=blue) markerattrs=(SIZE=5px SYMBOL=CircleFilled) ;
	reg x=Datum y=CO2_AVG_PPM_mon / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
	xaxis grid values=('01MAR1958'd to '31DEC2025'd by month);
	yaxis grid;
	where Datum >= "01MAR1958"d;
run;

/******************************************************************************/
/* Zeitreihen Diagramm für Monatsmittel CO2 PPM der letzten 5 Jahre im Detail */
/******************************************************************************/
proc sgplot data=work.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Monatliche Durchschnittswerte der CO2 PPM auf dem Mauna Loa, Hawaii";
	title2 height=08pt "Details der letzen 5 Jahre. Saisonale Schwankungen durch unstetes saisonales Pflanzenwachstum.";
	series x=Datum y=CO2_AVG_PPM_mon / markers
		markerfillattrs=(COLOR=blue) markerattrs=(SIZE=5px SYMBOL=CircleFilled) ;
	reg x=Datum y=CO2_AVG_PPM_mon / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
	xaxis grid;
	yaxis grid;
	where Datum >= intnx('YEAR', today(), -5, 'S');
run;

/********************************************************************/
/* Chart für Monatsmittel-Temperaturen monatlich in Deutschland     */
/********************************************************************/
footnote "Quelle: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/";

proc sgplot data=work.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Zeitreihe der monatlichen Mittel-Temperaturen (für Deutschland)";
	series x=Monat y=AVG_Temp_mon /  
		markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=5px SYMBOL=CircleFilled) lineattrs=(thickness=1) 
		dataskin=crisp legendlabel="Monatsmittel Temperatur";
	reg x=Monat y=AVG_Temp_mon / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
	refline 0 / lineattrs=(pattern=solid color=blue thickness=3px);
	refline 20 / lineattrs=(pattern=solid color=red thickness=3px);	
	xaxis grid display=(nolabel);
	yaxis grid;
	where Region = "Deutschland";
run;

/********************************************************************/
/* Chart für Monatsmittel-Temperaturen der Dekaden in Deutschland   */
/********************************************************************/

proc sgplot data=work.KLIMADATEN_MON_GRP_DEKADE;
	title height=14pt "Mittlere monatliche Dekaden-Temperaturen (für Deutschland)";
	series x=Monat_txt y=Mon_Temp_Mean / group=Dekade 
		markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled)  lineattrs=(thickness=1) 
		dataskin=crisp legendlabel="Monatsmittel Temperatur der Dekade";
	refline 0 / lineattrs=(pattern=solid color=blue thickness=3px);
	refline 15 / lineattrs=(pattern=solid color=red thickness=3px);	
	xaxis grid display=(nolabel);
	yaxis grid;
run;

/********************************************************************/
/* Zeitreihen Diagramm für Monatsmittel Temperaturen in Deutschland */
/********************************************************************/
proc sgplot data=work.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Monatliche Mittel-Temperaturen im Jahresvergleich (für Deutschland) ab 2010";
	series x=Monat_txt y=AVG_Temp_mon / group=Jahr_txt 
		markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled)  lineattrs=(thickness=1) 
		dataskin=crisp legendlabel="Monatsmittel Temperatur";
	refline 0 / lineattrs=(pattern=solid color=blue thickness=3px);
	refline 15 / lineattrs=(pattern=solid color=red thickness=3px);	
	xaxis grid display=(nolabel);
	yaxis grid;
	where Region = "Deutschland" and Jahr_txt > "2010";
run;

/*******************************************************************************/
/* Spannweite der monatlichen Mitteltemperaturen über die Jahre in Deutschland */
/*******************************************************************************/
PROC SGPLOT DATA=work.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Jährliche Spannweiten der Monats-Mittel-Temperaturen (für Deutschland)";
  VBOX AVG_Temp_mon / CATEGORY=Jahr_txt group=dekade capshape=serif connect=mean fill nooutliers;
  XAXIS grid LABEL="Jahr" ;
  YAXIS grid LABEL="Monatliche Mitteltemperatur (°C)";
  	where Region = "Deutschland" and AVG_Temp_year <> .;
RUN;

/********************************************************************/
/* Zeitreihen Diagramm für Jahresmittel Temperaturen in Deutschland */
/********************************************************************/
footnote "Quelle: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/air_temperature_mean/regional_averages_tm_year.txt";

proc sgplot data=WORK.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Jahresmittel-Temperaturen (für Deutschland)";
	series x=Jahr y=AVG_Temp_year / markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1) 
		datalabel=AVG_Temp_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Jahresmittel Temperatur";
	reg x=Jahr y=AVG_Temp_year / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
	xaxis grid display=(nolabel);
	yaxis grid min=6 max=11;
		where Region = "Deutschland" and AVG_Temp_year <> .;
run;

/********************************************************************/
/* Zeitreihen Diagramm für Anzahl Frosttage in Deutschland          */
/********************************************************************/

footnote "Quelle: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/frost_days/regional_averages_tnas_year.txt";

proc sgplot data=WORK.DWD_YEARLY_ALL;
	title height=14pt "Anzahl Frosttage (für Deutschland)";
	series x=Jahr y=Frosttage_year / markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1) 
		datalabel=Frosttage_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Anzahl Frosttage pro Jahr";
	reg x=Jahr y=Frosttage_year / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
    refline 100 / label="100 Frosttage" labelloc=inside labelpos=max
                  lineattrs=(pattern=solid color=blue thickness=3px pattern=shortdash);	
	xaxis grid display=(nolabel);
	yaxis grid min=0;
		where Region = "Deutschland" and Frosttage_year <> .;
run;


/********************************************************************/
/* Zeitreihen Diagramm für Anzahl Hitzetage in Deutschland          */
/********************************************************************/

footnote "Quelle: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/hot_days/regional_averages_txbs_year.txt";

proc sgplot data=WORK.DWD_YEARLY_ALL;
	title height=14pt "Anzahl Hitzetage über 30° (für Deutschland)";
	series x=Jahr y=Hitzetage_year / markers markerfillattrs=(COLOR=yellow) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1 color=red) 
		datalabel=Hitzetage_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Anzahl Hitzetage pro Jahr";
	reg x=Jahr y=Hitzetage_year / nomarkers lineattrs=(color=orange pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
        refline 10 / label="10 Hitzetage" labelloc=inside labelpos=min
                      lineattrs=(pattern=solid color=blue thickness=3px pattern=shortdash);	
	xaxis grid display=(nolabel);
	yaxis grid min=0;
		where Region = "Deutschland" and Hitzetage_year <> .;
run;


footnote "Quelle: tbd";

proc sgplot data=WORK.DWD_YEARLY_ALL;
	title height=14pt "Anzahl Frost- und Hitzetage (für Deutschland)";
	series x=Jahr y=Frosttage_year / markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1) 
		datalabel=Frosttage_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Anzahl Frosttage pro Jahr";
	reg x=Jahr y=Frosttage_year / nomarkers lineattrs=(color=red pattern=dash) 
		legendlabel="Trend (Frosttage)"
		degree=3 CLI;
	series x=Jahr y=Hitzetage_year / markers markerfillattrs=(color=green) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1 color=red) 
		datalabel=Hitzetage_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Anzahl Hitzetage pro Jahr";
	reg x=Jahr y=Hitzetage_year / nomarkers lineattrs=(color=orange pattern=dash) 
		legendlabel="Trend (Hitzetage)"
		degree=3 CLI;
	xaxis grid display=(nolabel);
	yaxis grid label="Tage pro Jahr";
		where Region = "Deutschland" and Hitzetage_year <> . and Frosttage_year <> .;
run;



/********************************************************************/
/* Zeitreihen Diagramm für Niederschlagsmengen in Deutschland       */
/********************************************************************/

footnote "Quelle: https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/annual/hot_days/regional_averages_txbs_year.txt";

proc sgplot data=WORK.DWD_YEARLY_ALL;
	title height=14pt "Niederschlagsmenge kumuliert jährlich in mm (für Deutschland)";
	series x=Jahr y=Niederschlag_year / markers markerfillattrs=(COLOR=blue) markerattrs=(SIZE=9px SYMBOL=CircleFilled) lineattrs=(thickness=1 color=lightblue) 
		datalabel=Niederschlag_year datalabelattrs=(size=8pt) dataskin=crisp 
		legendlabel="Anzahl Hitzetage pro Jahr";
	reg x=Jahr y=Niederschlag_year / nomarkers lineattrs=(color=blue pattern=dash) 
		legendlabel="Trend (Regression kubisch)"
		degree=3 CLI;
	xaxis grid display=(nolabel);
	yaxis grid label="Niederschlag pro Jahr in mm";
		where Region = "Deutschland" and Niederschlag_year <> .;
run;


/********************************************************************/
/* Chart für Jahresmittel-Temperaturen der Dekaden in Deutschland   */
/********************************************************************/
proc sgplot data=WORK.KLIMADATEN_MONATLICH_GESAMT;
	title height=14pt "Dekaden-Mitteltemperatur (für Deutschland)";
	title2 height=08pt "Y-Achse beginnt erst bei 6°";
	vbar dekade / response=AVG_Temp_year datalabel=AVG_Temp_year
		 fillattrs=(color=CX2470AD) fillType=gradient
		 stat=mean dataskin=crisp legendlabel="Dekadenmittel Temperatur";
	yaxis grid min=6 max=11;
		where Region = "Deutschland" and AVG_Temp_year <> .;
run;


/********************************************************************/
/* Chart für Tagesmittel-Temperaturen der Meeresoberfläche          */
/********************************************************************/
footnote "Quelle: https://climatereanalyzer.org/clim/sst_daily/json_2clim/oisst2.1_world2_sst_day.json";

ods graphics on / width=1400 height=1000;

proc sgplot data=work.world_ocean_temp_daily;
	title height=14pt "Tägliche Meeresoberfläche Temperatur 60°S-60°N, 0-360°E";
	title2 height=08pt "nach Tag des Jahres";
	series x=TagDesJahres y=AVG_Temp_day / group=Zeitraum
		/*markers markerfillattrs=(COLOR=orange) markerattrs=(SIZE=5px SYMBOL=CircleFilled) */
		dataskin=crisp legendlabel="Tagesmittel Temperatur";
	refline 20.25 / lineattrs=(pattern=solid color=red thickness=3px);	
	xaxis fitpolicy=thin display=(nolabel);
	yaxis grid;
	where Zeitraum in("2022", "2023", "2024", "2025", "2026", "1982-2010");
run;


ods graphics / reset;

*ods pdf close;

title;
footnote;



