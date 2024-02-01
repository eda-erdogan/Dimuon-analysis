// #include "ROOT/RDataFrame.hxx"

#include <TH1.h>

#include <TH1F.h>

#include <TStyle.h>

#include <TCanvas.h>

#include <TFile.h>

#include <TF1.h>

#include "TLegend.h"

#include <iostream>

double quadraticbackground(double *x, double *par)
{
    return par[0] + par[1] * x[0] + par[2] * x[0] * x[0]; //exponential func da denenebilir
}

// Lorenzian Peak function
double lorentzianPeak(double *x, double *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) /
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2])
   + .25*par[1]*par[1]);
}

// Sum of background and peak function
double fitFunction(double *x, double *par)
{
    double background = quadraticbackground(x, par);
    double lorentzPeak = lorentzianPeak(x, &par[3]);
    return background + lorentzPeak;
}
//JPSI FITTING

void makeplots()
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvasjpsi = new TCanvas("fitcanvasjpsi", "FitHist", 800, 600);
    fitcanvasjpsi->SetFillColor(15);
    fitcanvasjpsi->SetFrameFillColor(40);
    fitcanvasjpsi->SetGridx();
    fitcanvasjpsi->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(1, 5);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.8);
    histodimuonmass->SetStats(false);

   TF1 *fitJPsi = new TF1("fitJPsi", lorentzianPeak, 3.0 , double(3.20), 3);
    fitJPsi->SetLineColor(kGreen);
    fitJPsi->SetNpx(500);
    fitJPsi->SetParameters(1, 1, 3.1);
    histodimuonmass->Fit("fitJPsi", "R");

    //fitJPsi->SetParameter(0, 1000); // par [0] peak'in maks yüksekliği
    //fitJPsi->SetParameter(2.7, 3.4); // par [1] full width at half max
    //fitJPsi->SetParameter(2, 5.0); // par[2] peak'in konumu. bu üçlüyü kullanabilir miyim emin değilim
    //fitJPsi->Draw();
    //fitJPsi->Update();

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (2.50), double (3.50), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 3.1);
    //histodimuonmass->Fit("fitQuadraticBackground", "R");
   histodimuonmass->Fit("fitQuadraticBackground", "RQN"); 
    histodimuonmass->Fit("fitQuadraticBackground", "R"); 

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (1.900), double (3.800), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, 1, 1, 1, 1);
    histodimuonmass->Fit("fittotal", "RQN"); 
    histodimuonmass->Fit("fittotal", "R"); 





TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitJPsi,"Peak fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvasjpsi->SetGridx();
fitcanvasjpsi->SetGridy();

    
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitJPsi->Draw("same"); //aynı canvas üzerine çizdirmek için same kullandım
    fittotal->Draw("same");
    //legend->Draw();

    fitcanvasjpsi->Print("fitcanvasjpsi.pdf");
    fitcanvasjpsi->Update();
    histodimuonmass->Write();

} 
//FITETA, burada histogramın tamamını zoomlamadan fit'i görmek istediğim için yaptım
/* void makeplots()
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvaseta = new TCanvas("fitcanvaseta", "FitHist", 800, 600);
    fitcanvaseta->SetFillColor(15);
    fitcanvaseta->SetFrameFillColor(40);
    fitcanvaseta->SetGridx();
    fitcanvaseta->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(0.4398, 0.6697);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.8);
    histodimuonmass->SetStats(false);

    TF1 *fiteta = new TF1("fiteta", lorentzianPeak, double (0.4508), double(0.6597), 3);
    fiteta->SetLineColor(kCyan);
    fiteta->SetNpx(500);
    fiteta->SetParameters(1, 1, 0.5);
    histodimuonmass->Fit("fiteta", "R");

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (0.4600), double (0.6597), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
    

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (0.4600), double (0.6597), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, 1, 1, 1, 1);
    histodimuonmass->Fit("fittotal", "R");

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fiteta;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    fitcanvaseta->SetLogx();
    fitcanvaseta->SetLogy();

outf->cd();

    outf->WriteTObject(fiteta);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fiteta,"Peak fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvaseta->SetGridx();
fitcanvaseta->SetGridy();

    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fiteta->Draw("same"); //aynı canvas üzerine çizdirmek için same kullandım
    fittotal->Draw("same");
    legend->Draw();

    fitcanvaseta->SaveAs("fitcanvaseta.pdf");
    fitcanvaseta->Draw();


    file->Close();
    delete file;
    outf->Close();

}*/
//FIT PHI
/*
void makeplots()
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvasphi = new TCanvas("fitcanvasphi", "FitHist", 800, 600);
    fitcanvasphi->SetFillColor(15);
    fitcanvasphi->SetFrameFillColor(40);
    fitcanvasphi->SetGridx();
    fitcanvasphi->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(0.8795, 1.199);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.8);
    histodimuonmass->SetStats(false);

   TF1 *fitphi = new TF1("fitphi", lorentzianPeak, double (0.8994), double(1.150), 3);
    fitphi->SetLineColor(kGreen);
    fitphi->SetNpx(500);
    fitphi->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitphi", "R");

    // fitJPsi->SetParameter(0, 1000); // par [0] peak'in maks yüksekliği
    // fitJPsi->SetParameter(2.7, 3.4); // par [1] full width at half max
    // fitJPsi->SetParameter(2, 5.0); // par[2] peak'in konumu. bu üçlüyü kullanabilir miyim emin değilim
    //fitJPsi->Draw();
    //fitJPsi->Update();

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (0.8795), double (1.199), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (0.9094), double (1.150), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, 1, 1, 1, 1);
    histodimuonmass->Fit("fittotal", "R"); 

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fitphi;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");


outf->cd();

    outf->WriteTObject(fitphi);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitphi,"Peak fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvasphi->SetGridx();
fitcanvasphi->SetGridy();

    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitphi->Draw("same"); //aynı canvas üzerine çizdirmek için same kullandım
    fittotal->Draw("same");
    legend->Draw();

    fitcanvasphi->Print("fitcanvasphi.pdf");
    fitcanvasphi->Update();

    file->Close();
    delete file;
    outf->Close();
} */
/* //RHOANDOMEGA FITTING, uzak ölçekten baktım log set edip o zaman düzgün duruyor ama setrange'de background sınırları kötü

void makeplots() 
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvasrhoomega = new TCanvas("fitcanvasrhoomega", "FitHist", 800, 600);
    fitcanvasrhoomega->SetFillColor(15);
    fitcanvasrhoomega->SetFrameFillColor(40);
    fitcanvasrhoomega->SetGridx();
    fitcanvasrhoomega->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(0.600, 1.000);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.8);
    histodimuonmass->SetStats(false);

   TF1 *fitrhoomega = new TF1("fitrhoomega", lorentzianPeak, double (0.7300), double(0.8495), 3);
    fitrhoomega->SetLineColor(kGreen);
    fitrhoomega->SetNpx(500);
    fitrhoomega->SetParameters(0.1, 1, 1);
    histodimuonmass->Fit("fitrhoomega", "R");

    // fitrhoomega->SetParameter(0, 1000); // par [0] peak'in maks yüksekliği
    // fitrhoomega->SetParameter(2.7, 3.4); // par [1] full width at half max
    // fitrhoomega->SetParameter(2, 5.0); // par[2] peak'in konumu. bu üçlüyü kullanabilir miyim emin değilim
    //fitrhoomega->Draw();
    //fitrhoomega->Update();

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (0.6097), double (0.9094), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (0.600), double (1.000), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(0.1, 1, 1, 0.1, 1, 1);
    histodimuonmass->Fit("fittotal", "R"); 

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fitrhoomega;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    //fitcanvasrhoomega->SetLogx();
    //fitcanvasrhoomega->SetLogy(); 

outf->cd();

    outf->WriteTObject(fitrhoomega);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitrhoomega,"Peak fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitrhoomega->Draw("same"); 
    fittotal->Draw("same");
    legend->Draw();

    fitcanvasrhoomega->Print("fitcanvasrhoomega.pdf");
    fitcanvasrhoomega->Update();

    file->Close();
    delete file;
    outf->Close();
} */

//PHI FITING
/*
void makeplots()
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvasphi = new TCanvas("fitcanvasphi", "FitHist", 1000, 600);
    fitcanvasphi->SetFillColor(15);
    fitcanvasphi->SetFrameFillColor(40);
    fitcanvasphi->SetGridx();
    fitcanvasphi->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(0.800, 1.2000);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.8);
    histodimuonmass->SetStats(false);

   TF1 *fitphi = new TF1("fitphi", lorentzianPeak, double (0.9794), double(1.0600), 3);
    fitphi->SetLineColor(kGreen);
    fitphi->SetNpx(500);
    fitphi->SetParameters(0, 1, double(1.019));
    histodimuonmass->Fit("fitphi", "R");

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (0.8095), double (1.2090), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (0.8100), double (1.2100), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, double(1.019), 1, 1, double(1.019));
    histodimuonmass->Fit("fittotal", "R"); 

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fitphi;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    //fitcanvasphi->SetLogx();
    //fitcanvasphi->SetLogy(); 
outf->cd();

    outf->WriteTObject(fitphi);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitphi,"Peak fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvasphi->SetGridx();
fitcanvasphi->SetGridy();

    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitphi->Draw("same"); //aynı canvas üzerine çizdirmek için same kullandım
    fittotal->Draw("same");
    legend->Draw();

    fitcanvasphi->Print("fitcanvasphi.pdf");
    fitcanvasphi->Update();

    file->Close();
    delete file;
    outf->Close();
} */
//PSIPRIME FITTING, burada da yine zoom modunda signal çok iyi görünmüyor ama histogramın log halinde iyi bir fit alıyorum
/* void makeplots()
{

    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvaspsiprime = new TCanvas("fitcanvaspsiprime", "FitHist", 1000, 600);
    fitcanvaspsiprime->SetFillColor(kYellow-10);
    fitcanvaspsiprime->SetFrameFillColor(20);
    fitcanvaspsiprime->SetGridx();
    fitcanvaspsiprime->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(3.200, 4.200);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.1);
    histodimuonmass->SetStats(false);

   TF1 *fitpsiprime = new TF1("fitpsiprime", lorentzianPeak, double (3.500), double(3.850), 3);
    fitpsiprime->SetLineColor(kGreen);
    fitpsiprime->SetNpx(500);
    fitpsiprime->SetParameters(3.55, 3.8, 3.677);
    histodimuonmass->Fit("fitpsiprime", "R");

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (3.287), double (4.666), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (3.400), double (4.000), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(3.50, 3.8, 3.677, 3.50, 3, 3.677);
    histodimuonmass->Fit("fittotal", "R"); 

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fitpsiprime;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    //fitcanvaspsiprime->SetLogx();
    //fitcanvaspsiprime->SetLogy(); 
outf->cd();

    outf->WriteTObject(fitpsiprime);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitpsiprime,"Signal fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvaspsiprime->SetGridx();
fitcanvaspsiprime->SetGridy();


    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitpsiprime->Draw("same"); //aynı canvas üzerine çizdirmek için same kullandım
    fittotal->Draw("same");
    legend->Draw();

    fitcanvaspsiprime->Print("fitcanvaspsiprime.pdf");
    fitcanvaspsiprime->Update();

    file->Close();
    delete file;
    outf->Close();
} */
 //UPSILON FITTING, total hep en yüksek peak'e gitti. Bu aralıktaki upsilonlar muonlara decay eder mi?

/* void makeplots()
{
    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvasupsilon = new TCanvas("fitcanvasupsilon", "FitHist", 1000, 600);
    fitcanvasupsilon->SetFillColor(kYellow-10);
    fitcanvasupsilon->SetFrameFillColor(20);
    fitcanvasupsilon->SetGridx();
    fitcanvasupsilon->SetGridy();
    
    histodimuonmass->GetXaxis()->SetRangeUser(7.000, 14.000);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.1);
    histodimuonmass->SetStats(false);

   TF1 *fitupsilon = new TF1("fitupsilon", lorentzianPeak, double (8.500), double(11.000), 3);
    fitupsilon->SetLineColor(kGreen);
    fitupsilon->SetNpx(500);
    fitupsilon->SetParameters(1, 1, 9.887);
    histodimuonmass->Fit("fitupsilon", "R");

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (7.000), double (14.000), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 1);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (7.400), double (14.000), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, 9.000, 1, 1, 8.000);
    histodimuonmass->Fit("fittotal", "R"); 

    TFile *outf = new TFile("fits.root", "recreate");

    outf->Delete("fitupsilon;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    //fitcanvasupsilon->SetLogx();
    //fitcanvasupsilon->SetLogy(); 
outf->cd();

    outf->WriteTObject(fitupsilon);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitupsilon,"Signal fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvasupsilon->SetGridx();
fitcanvasupsilon->SetGridy();


    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitupsilon->Draw("same"); 
    fittotal->Draw("same");
    legend->Draw();

    fitcanvasupsilon->Print("fitcanvasupsilon.pdf");
    fitcanvasupsilon->Update();

    file->Close();
    delete file;
    outf->Close();
} 
/* // Z BOSON FITTING
void makeplots()
{
    TFile *file = TFile::Open("dimuonoutput.root");
    if (!file)
    {
        std::cout << "Error" << std::endl;
        return;
    }
    TH1F *histodimuonmass = (TH1F *)file->FindObjectAny("dimuon_mass");
    if (!histodimuonmass)
    {
        std::cout << "Error retrieving histogram" << std::endl;
        return;
    }
    TCanvas *fitcanvaszboson = new TCanvas("fitcanvaszboson", "FitHist", 1000, 600);
    fitcanvaszboson->SetFillColor(kYellow-10);
    fitcanvaszboson->SetFrameFillColor(20);
    fitcanvaszboson->SetGridx();
    fitcanvaszboson->SetGridy();
    
    //histodimuonmass->GetXaxis()->SetRangeUser(70.230, 143.300);
    histodimuonmass->SetMarkerStyle(15);
    histodimuonmass->SetMarkerSize(0.1);
    histodimuonmass->SetStats(false);

   TF1 *fitzboson = new TF1("fitzboson", lorentzianPeak, double (71.400), double(103.500), 3);
    fitzboson->SetLineColor(kGreen);
    fitzboson->SetNpx(500);
    fitzboson->SetParameters(77.17, 102.50, 90.79); //par0 ve par1 hiç fark yaratmadı
    histodimuonmass->Fit("fitzboson", "R");

    TF1 *fitQuadraticBackground = new TF1("fitQuadraticBackground", quadraticbackground, double (60.000), double (120.000), 3);
    fitQuadraticBackground->SetLineColor(kRed);
    fitQuadraticBackground->SetNpx(500);
    fitQuadraticBackground->SetParameters(1, 1, 70);
    histodimuonmass->Fit("fitQuadraticBackground", "R");
  

    TF1 *fittotal = new TF1("fittotal", fitFunction, double (66.840), double (133.400), 6);
    fittotal->SetLineColor(kBlue);
    fittotal->SetNpx(500);
    fittotal->SetParameters(1, 1, 1, 1, 1, 1);
    histodimuonmass->Fit("fittotal", "R"); //backgroundun ve totalin alt sınıra kadar inmesini engellemem gerekir mi?

    TFile *outf = new TFile("fits.root", "recreate"); // recreate yerine tüm çıktıları aynı dosyaya kaydedecek bir yöntem düşünemedim. Tree yaratmak olabilir mi?

    outf->Delete("fitzboson;1");
    outf->Delete("fitQuadraticBackground;1");
    outf->Delete("fittotal;1");

    fitcanvaszboson->SetLogx();
    fitcanvaszboson->SetLogy(); 
outf->cd();

    outf->WriteTObject(fitzboson);
    outf->WriteTObject(fitQuadraticBackground);
    outf->WriteTObject(fittotal);

TLegend *legend=new TLegend(0.7,0.7,0.88,0.88);
  legend->SetTextFont(60);
  legend->SetTextSize(0.025);
  legend->AddEntry(histodimuonmass,"Data","lpe");
  legend->AddEntry(fitQuadraticBackground,"Background fit","l");
  legend->AddEntry(fitzboson,"Signal fit","l");
  legend->AddEntry(fittotal,"Total Fit","l");

fitcanvaszboson->SetGridx();
fitcanvaszboson->SetGridy();


    histodimuonmass->Write();
    histodimuonmass->Draw();
    fitQuadraticBackground->Draw("same");
    fitzboson->Draw("same"); 
    fittotal->Draw("same");
    legend->Draw();

    fitcanvaszboson->Print("fitcanvaszboson.pdf");
    fitcanvaszboson->Update();

    file->Close();
    delete file;
    outf->Close();
} */


