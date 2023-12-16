#define dimuon_analysis_cxx
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RDataFrame.hxx>
#include "dimuon_analysis.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>

class TLorentzVector;

void dimuon_analysis::Loop()
{
   TFile *outf = new TFile("dimuonoutput.root", "recreate");
   // TH1F *h_muon_pt = new TH1F("h_muon_pt", "Muon Transverse Momentum; pt; Events", 30, 0, 3000);
   //  TH1F *h_muon_eta = new TH1F("h_muon_eta", "h_muon_eta", 5, -3, 3);
   //  TH1F *h_muon_phi = new TH1F("h_muon_phi", "h_muon_phi", 3000, -3.2, 3.2); //invariant mass dışında bunları ne için kullanabilirim?                      // -π to +π range
   TH1F *h_dimuon_mass = new TH1F("dimuon_mass", "Dimuon Mass; m_{#mu#mu}; Events", 30000, 0.25, 300);
   TCanvas *invariant_mass = new TCanvas("dimuoninvariantmass", "Dimuon Invariant Mass", 800, 700);

   if (fChain == 0)
      return;
   TLorentzVector p4_muon1(0, 0, 0, 0);
   TLorentzVector p4_muon2(0, 0, 0, 0); // başlangıç değerleri ne zaman değişir? ya da değişir mi?
   double pt, eta, phi, mass;
   double dimuon_mass;

   Long64_t nentries = fChain->GetEntriesFast();
   Long64_t nbytes = 0, nb = 0;

   int passedEvents = 0;

   for (Long64_t jentry = 0; jentry < nentries; jentry++)
   {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0)
         break;
      nb = fChain->GetEntry(jentry);
      nbytes += nb;

      if (jentry % 10000 == 0)
      {
         std::cout << "Processing event " << jentry << " of total number of events " << nentries << "!" << std::endl;
      }

      if (nMuon == 2 && Muon_charge[0] * Muon_charge[1] < 0)
      {
         for (int i_muon = 0; i_muon < nMuon; i_muon++)
         {
            pt = Muon_pt[i_muon];
            eta = Muon_eta[i_muon];
            phi = Muon_phi[i_muon]; // burada hatam phi olmayıp mass yazmasıydı ama yine de j/psi, eta, rho peakleri çıkmıştı. BUradan parametrelerin nasıl çalıştığını anlayabiliyor muyuz?
            mass = Muon_mass[i_muon];

            if (i_muon == 0)
            {
               p4_muon1.SetPtEtaPhiM(pt, eta, phi, mass);
            }
            else if (i_muon == 1)
            {
               p4_muon2.SetPtEtaPhiM(pt, eta, phi, mass);

            }
         }

         double dimuon_mass = (p4_muon1 + p4_muon2).M(); // bence burada bir düzenleme olmalı

         h_dimuon_mass->Fill(dimuon_mass);

         passedEvents++;
      }
   }
   std::cout << "Number of events passing the cuts: " << passedEvents << std::endl; // bunu başka şekilde nasıl eklerdim?
   // auto report = df.Report();

   invariant_mass->SetLogx();
   invariant_mass->SetLogy();

   outf->cd();
   h_dimuon_mass->Write();
   h_dimuon_mass->Draw();
   invariant_mass->cd();

   TLatex label;
   label.SetNDC(true);
   label.DrawLatex(0.175, 0.740, "#eta");
   label.DrawLatex(0.205, 0.775, "#rho,#omega");
   label.DrawLatex(0.270, 0.740, "#phi");
   label.DrawLatex(0.400, 0.800, "J/#psi");
   label.DrawLatex(0.415, 0.670, "#psi'");
   label.DrawLatex(0.485, 0.700, "Y(1,2,3S)");
   label.DrawLatex(0.755, 0.680, "Z");
   invariant_mass->SaveAs("invariantmass.pdf");
   invariant_mass->Draw(); 
   outf->Close();
}
