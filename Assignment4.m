%Assignment 4 Deep Learning in Data Science Harald Stiff
%This script implements a RNN that generates text similar to 
%the Harry Potter book "The Goblet of Fire".

%% Data Processing

book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars=unique(book_data);

[bsize1,bsize2]=size(book_chars);


char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i=1:bsize2
   char_to_ind(book_chars(i))=i;
   ind_to_char(i)=book_chars(i);  
end



%% TRAIN RNN
sig=0.1;
m=100;
K=length(book_chars);

RNN.b=zeros([m,1]);
RNN.c=zeros([K,1]);

RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;


[RNN_star,S,C]=GD(book_data,RNN,char_to_ind,ind_to_char);

 


%% Functions
 
 
 function [RNN_star,S,C] = GD(book_data,RNN,char_to_ind,ind_to_char)
    %GRADIENT DESCENT
    seq_length=25;
    iterations=10000;
    eta=0.1;
    eps=0.001
    epochs=3;
    for f = fieldnames(RNN)'
       m.(f{1})=zeros(size(RNN.(f{1}))); 
    end
    i=1;
    smooth_loss=ComputeLoss(chars2onehot(book_data(1:25),char_to_ind),chars2onehot(book_data(2:26),char_to_ind),RNN,zeros([100,1]));
    S=[smooth_loss];
    [a,h,o,y,p]=synth2(RNN,zeros([100,1]),chars2onehot('.',char_to_ind),200,0,0);
    chars=onehot2chars(y,ind_to_char);
    C=[chars];
    for j=1:epochs
        e=1
        h0=zeros([100,1]);
        while (e<length(book_data)-seq_length-1)
            x=book_data(e:e+seq_length);
            y=book_data(e+1:e+1+seq_length);
            x=chars2onehot(x,char_to_ind);
            y=chars2onehot(y,char_to_ind);
            [grads,hprev]=ComputeGradients(x,y,RNN,h0);
            for f = fieldnames(RNN)'
                m.(f{1})=m.(f{1})+grads.(f{1}).^2;
                RNN.(f{1}) = RNN.(f{1})-eta./(sqrt(m.(f{1})+eps)).*grads.(f{1});
            end
            e=e+seq_length;
            h0=hprev;
            i=i+1;
            cost=ComputeLoss(x,y,RNN,h0);
            smooth_loss=0.999*smooth_loss+cost*0.001;
            S=[S,smooth_loss];
            if mod(i,500)==0
               smooth_loss
               [a,h,o,y,p]=synth2(RNN,hprev,x(:,1),200,0,0);
               chars=onehot2chars(y,ind_to_char)
               i
               S=[S,smooth_loss];
            end
            if mod(i,10000)==0
               [a,h,o,y,p]=synth2(RNN,hprev,x(:,1),200,0,0);
               chars=onehot2chars(y,ind_to_char);
               C=[C;chars];
               
            end
        end
    end
    RNN_star=RNN;
 
 
 end
 
 
 function onehot=chars2onehot(chars,char_to_ind)
    %CHARACTERS TO ONEHOT-MATRIX
    onehot=zeros([83,length(chars)]);
    for i=1:length(chars)
        onehot(char_to_ind(chars(i)),i)=1;
    end
 end
 
 function characters=onehot2chars(onehot,ind_to_char)
    %ONEHOT-MATRIX TO CHARACTERS
    [n,m]=size(onehot);
    characters=[];
    for i=1:m
       characters=[characters, ind_to_char(find(onehot(:,i)>0))]; 
    end
 end
 
 
 function [dldh,dlda]= dl(a,h,o,y,P,g,RNN,seq_length)
    %GRADIENTS NEEDED FOR OTHER GRADIENT CALCULATIONS
    dldhtau=g(end,:)*RNN.V;
    dldatau=dldhtau*diag(1-tanh(a(:,end)).^2);
    size(g);
    dldh=[dldhtau];
    dlda=[dldatau];
    seq_length;
    for i=1:seq_length-1
        dldht=g(end-i,:)*RNN.V+dlda(1,:)*RNN.W;
        dldh=[dldht;dldh];
        dldat=dldh(1,:)*diag(1-tanh(a(:,end-i)).^2);
        dlda=[dldat;dlda];
    end
 end
 
 function [grad,h_end]=ComputeGradients(X,Y,RNN,h0)
    %GRADIENTS OF RNN
    [a,h,o,y,P]=synth(RNN,X,h0);
    [n,m]=size(X);
    g=-(Y-P)';
    grad.V = 0;
    grad.W = 0;
    grad.U = 0;
    grad.b = 0;
    grad.c = 0;
    [dldh,g2]=dl(a,h,o,y,P,g,RNN,m);
    for i=1:m
       grad.V = grad.V + g(i,:)'*h(:,i+1)';
       
       grad.c = grad.c + g(i,:)';
       
       grad.W = grad.W + g2(i,:)'*h(:,i)';
       
       grad.b = grad.b + g2(i,:)';
       
       grad.U = grad.U + g2(i,:)'*X(:,i)'; 
    end
    for f = fieldnames(RNN)'
        grad.(f{1})=max(min(grad.(f{1}), 5), -5);
    end
    h_end=h(:,end);
 end
 
 function [cost] = ComputeLoss(X,Y,RNN,h0)
    %LOSS
    [a,h,o,y,P]=synth(RNN,X,h0);
    [n,m]=size(Y);
    cost=0;
    for i = 1:m
        cost=cost-log(Y(:,i)'*P(:,i));
    end
    
 end
 
 function M = seq2mat(X)
    l=length(X);
    M=zeros([83,l]);
    for i = 1:l
       M(X(i),i)=1; 
    end
 end
 
 function [P]= softmax(X)
 %SOFTMAX
    [a1,b]=size(X);
    P=zeros(size(X));
    for i=1:b
        P(:,i)=exp(X(:,i))/(sum(exp(X(:,i))));
    end
 end
 
 
 function [a,h,o,y,p]= synth2(RNN,h0,x0,n,train,x)
 %SYNTHESIZE
    a=[];
    h=[h0];
    p=[];
    o=[];
    y=zeros([83,n]);
    x_next=x0;
    if train == 1
        [a1,a2]=size(x);
        n=a2;
    end
    for i=1:n
       if train == 1
          x_next= x(:,i);
       end
       at=RNN.W*h(:,end)+RNN.U*x_next+RNN.b;
       a=[a,at];

       ht=tanh(a(:,end));
       h=[h,ht];
       
       ot=RNN.V*h(:,end)+RNN.c;
       o=[o,ot];
     
       pt=softmax(o(:,end));
       p=[p,pt];
       
       cp = cumsum(p(:,end));
       a1 = rand;
       ixs = find(cp-a1 >0);
       ii = ixs(1);
       y(ii,i)=1;

       x_next=y(:,i);
    end
 
 
 end
 
 
 function [a,h,o,y,p] =synth(RNN,x,h0)
 %SYNTHESIZE
    a=[];
    h=[h0];
    p=[];
    o=[];
    y=[];
    [n,m]=size(x);
    for i=1:m
       at=RNN.W*h(:,end)+RNN.U*x(:,i)+RNN.b;
       a=[a,at];
       
       ht=tanh(a(:,end));
       h=[h,ht];
       
       ot=RNN.V*h(:,end)+RNN.c;
       o=[o,ot];
     
       pt=softmax(o(:,end));
       p=[p,pt];

    end
 end
 
 
 
 function num_grads = ComputeGradsNum(X, Y, RNN, h)

    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        f{1};
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
 end
function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;

        RNN_try.(f)(i) = RNN.(f)(i) - h;

        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end