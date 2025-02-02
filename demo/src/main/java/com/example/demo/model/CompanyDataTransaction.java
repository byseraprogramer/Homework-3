package com.example.demo.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.*;
import lombok.*;

import java.io.Serializable;
import java.util.Date;

//@Data
@Entity
@Table(name = "CompanyDataTransaction")
@JsonIgnoreProperties({"companydata"})
public class CompanyDataTransaction implements Serializable {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne
    private CompanyData companydata;

    private String date;

    private String lastPrice;

    private String min;

    private String max;

    private String averagePrice;

    private String percentageChange;

    private String quantity;

    private String turnover;

    private String totalTurnover;

    private String sma10;

    private String ema10;

    private String rsi;

    private String cci;

    private String k;

    private String d;

    private String signal;

    private String CCI_signal;

    private String Stochastic_signal;

    private String SMA_20;

    private String EMA_20;

    private String WMA_10;


    public CompanyDataTransaction(CompanyData companydata, String date, String lastPrice, String min, String max, String averagePrice, String percentageChange, String quantity, String turnover, String totalTurnover, String sma10, String ema10, String rsi, String cci, String k, String d, String signal, String CCI_signal, String stochastic_signal, String EMA_20, String WMA_10, String SMA_20) {
        this.companydata = companydata;
        this.date = date;
        this.lastPrice = lastPrice;
        this.min = min;
        this.max = max;
        this.averagePrice = averagePrice;
        this.percentageChange = percentageChange;
        this.quantity = quantity;
        this.turnover = turnover;
        this.totalTurnover = totalTurnover;
        this.sma10 = sma10;
        this.ema10 = ema10;
        this.rsi = rsi;
        this.cci = cci;
        this.k = k;
        this.d = d;
        this.EMA_20 = EMA_20;
        this.WMA_10 = WMA_10;
        this.SMA_20 = SMA_20;
        this.signal = signal;
        this.CCI_signal = CCI_signal;
        Stochastic_signal = stochastic_signal;
    }

    public CompanyDataTransaction() {}

    public String getLastPrice() {
        return lastPrice;
    }

    public void setLastPrice(String lastPrice) {
        this.lastPrice = lastPrice;
    }

    public String getMin() {
        return min;
    }

    public void setMin(String min) {
        this.min = min;
    }

    public String getMax() {
        return max;
    }

    public void setMax(String max) {
        this.max = max;
    }

    public String getAveragePrice() {
        return averagePrice;
    }

    public void setAveragePrice(String averagePrice) {
        this.averagePrice = averagePrice;
    }

    public String getPercentageChange() {
        return percentageChange;
    }

    public void setPercentageChange(String percentageChange) {
        this.percentageChange = percentageChange;
    }

    public String getQuantity() {
        return quantity;
    }

    public void setQuantity(String quantity) {
        this.quantity = quantity;
    }

    public String getTurnover() {
        return turnover;
    }

    public void setTurnover(String turnover) {
        this.turnover = turnover;
    }

    public String getTotalTurnover() {
        return totalTurnover;
    }

    public void setTotalTurnover(String totalTurnover) {
        this.totalTurnover = totalTurnover;
    }

    public String getSma10() {
        return sma10;
    }

    public void setSma10(String sma10) {
        this.sma10 = sma10;
    }

    public String getEma10() {
        return ema10;
    }

    public void setEma10(String ema10) {
        this.ema10 = ema10;
    }

    public String getRsi() {
        return rsi;
    }

    public void setRsi(String rsi) {
        this.rsi = rsi;
    }

    public String getCci() {
        return cci;
    }

    public void setCci(String cci) {
        this.cci = cci;
    }

    public String getK() {
        return k;
    }

    public void setK(String k) {
        this.k = k;
    }

    public String getD() {
        return d;
    }

    public void setD(String d) {
        this.d = d;
    }

    public String getSignal() {
        return signal;
    }

    public void setSignal(String signal) {
        this.signal = signal;
    }

    public String getCCI_signal() {
        return CCI_signal;
    }

    public void setCCI_signal(String CCI_signal) {
        this.CCI_signal = CCI_signal;
    }

    public String getStochastic_signal() {
        return Stochastic_signal;
    }

    public void setStochastic_signal(String stochastic_signal) {
        Stochastic_signal = stochastic_signal;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public CompanyData getCompanydata() {
        return companydata;
    }

    public void setCompanydata(CompanyData companydata) {
        this.companydata = companydata;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getSMA_20() {
        return SMA_20;
    }

    public void setSMA_20(String SMA_20) {
        this.SMA_20 = SMA_20;
    }

    public String getEMA_20() {
        return EMA_20;
    }

    public void setEMA_20(String EMA_20) {
        this.EMA_20 = EMA_20;
    }

    public String getWMA_10() {
        return WMA_10;
    }

    public void setWMA_10(String WMA_10) {
        this.WMA_10 = WMA_10;
    }
}