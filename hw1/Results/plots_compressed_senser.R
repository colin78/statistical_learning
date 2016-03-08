phase_L0 = as.matrix(read.csv("phase_L0.csv", header=FALSE))
phase_L1 = as.matrix(read.csv("phase_L1.csv", header=FALSE))

n = 20
df_L0 = data.frame(m_n=1/n,k_m=1,recovered=TRUE)
df_L1 = data.frame(m_n=1/n,k_m=1,recovered=TRUE)

for (m in c(2:n))
{
  for (k in c(1:m))
  {
    if(phase_L0[m,k]==1){
      df_L0 = rbind(df_L0, c(m/n,k/m,TRUE))
    } else {
      df_L0 = rbind(df_L0, c(m/n,k/m,FALSE))
    }
    if(phase_L1[m,k]==1){
      df_L1 = rbind(df_L1, c(m/n,k/m,TRUE))
    }else{
      df_L1 = rbind(df_L1, c(m/n,k/m,FALSE))
    }
  }
}

df_L0$recovered <- factor(df_L0$recovered, rev(levels(factor(df_L0$recovered))))
df_L1$recovered <- factor(df_L1$recovered, rev(levels(factor(df_L1$recovered))))

library(ggplot2)
g1 = ggplot(df_L0, aes(m_n,k_m)) + geom_point(aes(color=factor(recovered))) +
  scale_color_manual(name="Recovered", values=c("blue","purple"), labels=c("True", "Oversparsity")) + 
  ggtitle("Phase Diagram for MIO problem") +
  labs(x="m/n", y="k/m")
g2 = ggplot(df_L1, aes(m_n,k_m)) + geom_point(aes(color=factor(recovered))) +
  scale_color_manual(name="Recovered", values=c("blue","red"), labels=c("True", "False")) + 
  ggtitle("Phase Diagram for LO problem") +
  labs(x="m/n", y="k/m")

ggsave("phase_MIO.pdf", g1)
ggsave("phase_LO.pdf", g2)


