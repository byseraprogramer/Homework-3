package com.example.demo.model;

import com.fasterxml.jackson.annotation.JsonManagedReference;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

//@Data
@Entity
@Table(name = "CompanyData")
public class CompanyData {
    @Id
    private String companyCode;

    @OneToMany(mappedBy = "companydata")
    @JsonManagedReference
    List<CompanyDataTransaction> transactions;

    public CompanyData(String companyCode) {
        this.companyCode = companyCode;
        transactions = new ArrayList<>();
    }

    public CompanyData() {}

    public String getCompanyCode() {
        return companyCode;
    }

    public List<CompanyDataTransaction> getTransactions() {
        return transactions;
    }

    public CompanyDataTransaction addTransaction(CompanyDataTransaction transaction) {
        transactions.add(transaction);
        return transaction;
    }

    public void removeTransaction(CompanyDataTransaction transaction) {
        transactions.remove(transaction);
    }

    public void setCode(String companyCode) {
        this.companyCode = companyCode;
    }
}